#!/usr/bin/env python3
import argparse
import os
import json
import numpy as np
from numpy.linalg import norm
from pathlib import Path
import sys
from scipy.signal import find_peaks
from skimage.exposure import is_low_contrast
from skimage.util.dtype import dtype_range, dtype_limits
from PIL import ImageDraw, Image
from torchvision.utils import draw_segmentation_masks
import torch
import cv2
import math
from scipy.stats import beta
import pickle
import lzma

parser = argparse.ArgumentParser(prog='torch_process_segm.py', description='Output image mask with possible road centres marked')
parser.add_argument('filename', metavar='FILENAME', help='Saved numpy (.npz or .npy) file to process, or list of such files (see -F)')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Run in verbose mode')
parser.add_argument('--filelist', '-F', action='store_true', default=False, help='Supplied path is actually a list of numpy filenames, one per line, to process.')
parser.add_argument('--fast', action='store_true', default=False, help='Fast mode, skip most functionality except: Road Finding, SKImage Contrast, Tone-mapping, and panoramic-image cropping.')
parser.add_argument('--centres-only', action='store_true', default=False, help='Skip all functionality except Road Finding')
parser.add_argument('--overwrite', '-O', action='store_true', default=False, help='Overwrite output files')
parser.add_argument('--sqloutdir', '-S', metavar='DIR', default=None, help='Directory to output SQL files')
parser.add_argument('--cropsdir', metavar='DIR', default=None, help='Directory to output cropped JPGs (default: same directory as original image)')
parser.add_argument('--tiles', '-T', metavar='FILENAME-OR-DIR', required=True, help='Directory to find tiles files in JSON, or a tiles picklefile')
parser.add_argument('--dirprefix', '-D', default='/data/img/mapillary', help='prefix of system path for images')
parser.add_argument('--urlprefix', '-U', default='/img/mapillary', help='prefix of URL for images')
parser.add_argument('--cityname', '-C', default='Amsterdam', help='name of city associated with the given numpy files')
parser.add_argument('--log', action='store_true', default=False, help='Save verbose output to .out file')
parser.add_argument('--blur', action='store_true', default=False, help='Run Gaussian blur before finding edges')
parser.add_argument('--palette-file', '-P', metavar='FILENAME', default=None, help='File with list of colour names for mask output, one per line')
parser.add_argument('--mask-alpha', metavar='ALPHA', default=0.7, type=float, help='Alpha transparency value when drawing mask over image (0 = fully transparent; 1 = fully opaque)')
parser.add_argument('--no-houghtransform-road-centrelines', action='store_true', default=False, help='Do not draw Hough transform-based road centrelines')
parser.add_argument('--no-segmentation-road-centrelines', action='store_true', default=False, help='Do not draw segmentation-based road centrelines')
parser.add_argument('--maskfile', '-m', nargs='?', metavar='FILENAME', default=None, const=True, help='Output filename for mask image')
parser.add_argument('--plusfile', '-p', metavar='FILENAME', default=None, help='Output filename for "plus" image; with the leftmost 25-percent appended to the righthand side')
parser.add_argument('--overfile', '-o', nargs='?', metavar='FILENAME', default=None, const=True, help='Output filename for image with centrelines drawn over')
parser.add_argument('--edgefile', '-e', metavar='FILENAME', default=None, help='Output filename for edges image')
parser.add_argument('--linefile', '-l', metavar='FILENAME', default=None, help='Output filename for lines image')
parser.add_argument('--blurfile', '-B', metavar='FILENAME', default=None, help='Output filename for Gaussian blurred image')
parser.add_argument('--blobfile', '-b', metavar='FILENAME', default=None, help='Output filename for blobs image')
parser.add_argument('--dataset', metavar='DATASET', default=None, help='Override segmentation dataset name (for visualisation)')
parser.add_argument('--road-peaks-distance', metavar='N', default=None, type=int, help='Distance between peaks of road pixels')
parser.add_argument('--road-peaks-prominence', metavar='N', default=None, type=int, help='Prominence of peaks of road pixels')
parser.add_argument('--houghlines-rho', metavar='RHO', default=None, type=float, help='Hough transform RHO parameter')
parser.add_argument('--houghlines-theta', metavar='THETA', default=None, type=float, help='Hough transform THETA parameter')
parser.add_argument('--houghlines-threshold', metavar='THRESH', default=None, type=int, help='Hough transform THRESHOLD parameter')
parser.add_argument('--houghlines-min-theta', metavar='THETA', default=None, type=float, help='Hough transform MIN_THETA parameter')
parser.add_argument('--houghlines-max-theta', metavar='THETA', default=None, type=float, help='Hough transform MAX_THETA parameter')

# https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image
def rms_contrast(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grey.std()

def michaelson_contrast(img):
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

    # compute min and max of Y
    min = np.min(Y).astype(float)
    max = np.max(Y).astype(float)

    # compute contrast
    contrast = (max-min)/(max+min)
    return contrast

###################################################
# https://towardsdatascience.com/measuring-enhancing-image-quality-attributes-234b0f250e10
RED_SENSITIVITY = 0.299
GREEN_SENSITIVITY = 0.587
BLUE_SENSITIVITY = 0.114
def convert_to_brightness_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        raise ValueError("uint8 is not a good dtype for the image")

    return np.sqrt(
        image[..., 2] ** 2 * RED_SENSITIVITY
        + image[..., 1] ** 2 * GREEN_SENSITIVITY
        + image[..., 0] ** 2 * BLUE_SENSITIVITY
    )
def get_resolution(image: np.ndarray):
    height, width = image.shape[:2]
    return height * width

def brightness_histogram(image: np.ndarray) -> np.ndarray:
    nr_of_pixels = get_resolution(image)
    brightness_image = convert_to_brightness_image(image)
    hist, _ = np.histogram(brightness_image, bins=256, range=(0, 255))
    return hist / nr_of_pixels
def distribution_pmf(dist, start, stop, nr_of_steps):
    xs = np.linspace(start, stop, nr_of_steps)
    ys = dist.pdf(xs)
    # divide by the sum to make a probability mass function
    return ys / np.sum(ys)
def correlation_distance(
    distribution_a: np.ndarray, distribution_b: np.ndarray
) -> float:
    dot_product = np.dot(distribution_a, distribution_b)
    squared_dist_a = np.sum(distribution_a ** 2)
    squared_dist_b = np.sum(distribution_b ** 2)
    return dot_product / math.sqrt(squared_dist_a * squared_dist_b)
def compute_hdr(cv_image: np.ndarray):
    img_brightness_pmf = brightness_histogram(np.float32(cv_image))
    ref_pmf = distribution_pmf(beta(2, 2), 0, 1, 256)
    return correlation_distance(ref_pmf, img_brightness_pmf)
###################################################

# Taken from skimage source code:
def skimage_contrast(image, lower_percentile=1, upper_percentile=99):
    image = np.asanyarray(image)

    if image.dtype == bool:
        return not ((image.max() == 1) and (image.min() == 0))

    if image.ndim == 3:
        from skimage.color import rgb2gray, rgba2rgb  # avoid circular import

        if image.shape[2] == 4:
            image = rgba2rgb(image)
        if image.shape[2] == 3:
            image = rgb2gray(image)

    dlimits = dtype_limits(image, clip_negative=False)
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / (dlimits[1] - dlimits[0])
    return ratio

# https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

# https://stackoverflow.com/questions/14243472/estimate-brightness-of-an-image-opencv
def simple_brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

# http://alienryderflex.com/hsp.html
def finley_brightness(img):
    return np.average(np.sqrt(np.sum(img.reshape(-1,3).astype(np.float32)**2 * [0.114, 0.587, 0.299], axis=1)))

def laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

##################################################

# Given a matrix of predicted labels for pixel segmentation, where the label
# for 'road' is 0, return the count of the longest run of 'road' labeled pixels
# in the bottom part of the input matrix.
def road_pixels_per_col(pred):
    h2 = pred.shape[0]//2
    # a is a boolean matrix (0 meaning non-road and 1 meaning road)
    a = pred == 0.0
    out = np.zeros(a.shape[1])
    # For each column, i
    for i in range(a.shape[1]):
      # RLE - Run Length Encode column i in the matrix a.
      # z = [count of value v]
      # p = [position of first v]
      # v = [actual value]
      (z, p, v) = rle(a[:,i])
      # For each column, test for runs of 'True' (v != 0) in the bottom part of
      # the image (p > h2), and choose the largest run (z):
      out[i]=z[np.logical_and(p > h2, v != 0)].max(initial=0)
    return out

# Given a matrix of predicted labels for pixel segmentation, where the label
# for 'road' is 0, return an array corresponding to the vertical distance
# between the bottom of the image and the topmost 'road' pixel in each column
# of the input matrix.
def road_pixel_dist_from_bottom(pred):
    h2 = pred.shape[0] // 2
    # a is a boolean matrix (0 meaning non-road and 1 meaning road)
    a = pred == 0.0
    out = np.zeros(a.shape[1])
    # For each column, i
    for i in range(a.shape[1]):
        # Find the rows (js) with road pixels in the bottom half of the image.
        js = np.argwhere(a[h2:,i] != 0)
        # Distance from image bottom to topmost road pixel, or 0 if no road.
        out[i] = h2 - js[0,0] if np.any(js) else 0
    return out

# Given a matrix of predicted labels for pixel segmentation, where the label
# for 'road' is 0, return an array containing the X-coordinate values that
# identify the centrelines of roads in the image corresponding to the
# segmentation array.
def road_centres(pred, distance=2000, prominence=100):
    road = road_pixel_dist_from_bottom(pred)
    rppc = road_pixels_per_col(pred)
    road += rppc / 8
    # Adding padding on either side to ensure find_peaks will find peaks near
    # the edges.
    padding = prominence*2
    roadplus = np.concatenate((np.zeros(padding),road,np.zeros(padding)))
    peaks = find_peaks(roadplus,distance=distance,prominence=prominence)[0]
    return peaks - padding

##################################################

# image: PIL image
# segmentation: Torch matrix
# road_centres: Matrix columns corresponding to road centres found.
#               It is permitted for road centres to exceed matrix width by 25%
#               because detection is performed on a matrix that wraps the first
#               quarter of the matrix onto the righthand side in order not to
#               miss any roads that are on the edge.
def crop_panoramic_image(image, segmentation, road_centres):
    # img*: variables in image coordinates
    imgw, imgh = image.size
    # Segmentation may be performed on a downscaled image
    # mat*: variables in matrix coordinates
    matw = segmentation.size(dim=1)
    # segmentation may have bottom cropped off, therefore calculate 'matrix
    # height' based on image & assume that bottom quarter of segmentation
    # matrix is not accessible
    math = imgh * matw // imgw

    # In addition to road centres we are also interested in views looking
    # slightly to the left and right of the road, thus including more of the
    # surroundings.
    matxset = set()
    matw4 = matw // 4
    matw8 = matw // 8
    matw4_100 = matw4 // 100
    math4 = math // 4
    mathFor43 = int(matw4 * 3 / 4) # height for 4:3 ratio submatrix
    # assumes panoramic image has width approximately 2x the height
    assert(mathFor43 <= math//2)
    # road_centres is in matrix coords:
    for centre in road_centres:
        matxleft = centre - matw8 + matw4_100
        matxright = centre + matw8 - matw4_100
        if matxleft < 0: matxleft += matw
        elif matxright >= matw: matxright -= matw
        matxset.add(matxleft)
        matxset.add(centre)
        matxset.add(matxright)

    # Calculate dimensions and offsets
    # The size of a cropped image will be: (w4, hFor43)
    # The crop will start at offset (x - w4/2, h4) for each x in the xlist
    #
    # Some complications arise when the desired cropped image wraps around
    # either end of the panoramic input image, or because image segmentation
    # input is calculated based on an image that has already been wrapped with
    # the first 25% of the image copied to the right-hand side (to ensure that
    # any roads on the edges of the panoramic image are found).
    imgw4 = int(imgw / 4)
    imgw8 = int(imgw / 8)
    imgh4 = int(imgh / 4)
    imghFor43 = int(imgw4 * 3 / 4) # height for 4:3 ratio image
    imgw98 = imgw + imgw8
    imgxwrapneeded = int(imgw * 7 / 8)

    images = []
    subsegms = []
    infos = []

    # Crop the panoramic image based on road centers
    for matx in matxset:
        imgx = imgw * matx // matw  # img coords
        # Thanks to Ilse Abril Vázquez Sánchez for translating my original
        # shell script code (process_out_file.sh) into Python.
        #
        # Wrapped all the way around:
        if imgx >= imgw98:
            matwrapx = matx - matw
            imgwrapx = imgx - imgw
            matxlo = int(matwrapx - matw8)
            imgxlo = int(imgwrapx - imgw8)
            cropped_image = image.crop((imgxlo, imgh4, imgxlo + imgw4, imgh4 + imghFor43))
            cropped_segmentation = segmentation[math4:math4+mathFor43, matxlo:matxlo+matw4]
            info = {
                'imgx': imgx, 'matx': matx, 'case': 'imgx >= imgw98', 'vars': { 'matxlo': matxlo, 'imgxlo': imgxlo },
                'imgpieces': [ (imgxlo, imgh4, imgxlo + imgw4, imgh4 + imghFor43) ],
                'matpieces': [ (math4,math4+mathFor43, matxlo,matxlo+matw4) ]
            }
        
        # Cropped image requires assembly of two sides, wrapping around
        # righthand side of image:
        elif imgx > imgxwrapneeded:
            matxlo = int(matx - matw8)
            imgxlo = int(imgx - imgw8)
            # width of piece 1: between xlo and the righthand side of the image
            matw4_p1 = matw - matxlo
            imgw4_p1 = imgw - imgxlo
            # width of piece 2: the remaining width needed, starting from the
            # lefthand side of the image
            matw4_p2 = matw4 - matw4_p1
            imgw4_p2 = imgw4 - imgw4_p1

            # Crop and concatenate image and segmentation
            cropped_image_1 = image.crop((imgxlo, imgh4, imgxlo + imgw4_p1, imgh4 + imghFor43))
            cropped_image_2 = image.crop((0, imgh4, imgw4_p2, imgh4 + imghFor43))

            cropped_image = Image.new(image.mode, (imgw4, imghFor43))
            cropped_image.paste(cropped_image_1, (0, 0))
            cropped_image.paste(cropped_image_2, (imgw4_p1, 0))

            cropped_segmentation_1 = segmentation[math4:math4+mathFor43, matxlo:matxlo+matw4_p1]
            cropped_segmentation_2 = segmentation[math4:math4+mathFor43, 0:matw4_p2]
            cropped_segmentation = torch.cat((cropped_segmentation_1, cropped_segmentation_2), dim=1)

            info = {
                'imgx': imgx, 'matx': matx, 'case': 'imgx > imgxwrapneeded',
                'vars': {
                    'matxlo': matxlo, 'imgxlo': imgxlo,
                    'matw4_p1': matw4_p1, 'imgw4_p1': imgw4_p1,
                    'matw4_p2': matw4_p2, 'imgw4_p2': imgw4_p2,

                },
                'imgpieces': [ (imgxlo, imgh4, imgxlo + imgw4_p1, imgh4 + imghFor43), (0, imgh4, imgw4_p2, imgh4 + imghFor43) ],
                'matpieces': [  (math4,math4+mathFor43, matxlo,matxlo+matw4_p1), (math4,math4+mathFor43, 0,matw4_p2) ]
            }
        
        # Cropped image requires assembly of two sides, wrapping around
        # lefthand side of image:
        elif imgx < imgw8:
            # given x appears between 0 and w8:
            # x coords:        0     x    w8
            #            |<---------w4----------->|
            #            |<----w8--->|<----w8---->|
            #                  |<-x->|
            #            |<--->|<---------------->|
            #              p1          p2
            # ergo, width of piece 1 (p1) = w8 - x
            imgw4_p1 = int(imgw8 - imgx)
            matw4_p1 = int(matw8 - matx)
            imgxhi = imgw - imgw4_p1
            matxhi = matw - matw4_p1
            # width of piece 2 (p2) is remainder
            imgw4_p2 = imgw4 - imgw4_p1
            matw4_p2 = matw4 - matw4_p1

            # Crop and concatenate image and segmentation
            cropped_image_1 = image.crop((imgxhi, imgh4, imgxhi + imgw4_p1, imgh4 + imghFor43))
            cropped_image_2 = image.crop((0, imgh4, imgw4_p2, imgh4 + imghFor43))

            cropped_image = Image.new(image.mode, (imgw4, imghFor43))
            cropped_image.paste(cropped_image_1, (0, 0))
            cropped_image.paste(cropped_image_2, (imgw4_p1, 0))

            cropped_segmentation_1 = segmentation[math4:math4+mathFor43, matxhi:matxhi+matw4_p1]
            cropped_segmentation_2 = segmentation[math4:math4+mathFor43, 0:matw4_p2]
            cropped_segmentation = torch.cat((cropped_segmentation_1, cropped_segmentation_2), dim=1)

            info = {
                'imgx': imgx, 'matx': matx, 'case': 'imgx < imgw8',
                'vars': {
                    'matxhi': matxhi, 'imgxhi': imgxhi,
                    'matw4_p1': matw4_p1, 'imgw4_p1': imgw4_p1,
                    'matw4_p2': matw4_p2, 'imgw4_p2': imgw4_p2,

                },
                'imgpieces': [ (imgxhi, imgh4, imgxhi + imgw4_p1, imgh4 + imghFor43), (0, imgh4, imgw4_p2, imgh4 + imghFor43) ],
                'matpieces': [ (math4,math4+mathFor43, matxhi,matxhi+matw4_p1), (math4,math4+mathFor43, 0,matw4_p2)]
            }
        # Straightforward crop
        else:
            matxlo = int(matx - matw8)
            imgxlo = int(imgx - imgw8)
            cropped_image = image.crop((imgxlo, imgh4, imgxlo + imgw4, imgh4 + imghFor43))
            cropped_segmentation = segmentation[math4:math4+mathFor43, matxlo:matxlo+matw4]
            info = {
                'imgx': imgx, 'matx': matx, 'case': 'default',
                'vars': { 'matxlo': matxlo, 'imgxlo': imgxlo },
                'imgpieces': [ (imgxlo, imgh4, imgxlo + imgw4, imgh4 + imghFor43) ],
                'matpieces': [ (math4,math4+mathFor43, matxlo,matxlo+matw4) ]
            }

        images.append(cropped_image)
        subsegms.append(cropped_segmentation)
        infos.append(info)

    return images, subsegms, infos

##################################################

def main():
    args = parser.parse_args()
    def vlog(s):
        if args.verbose:
            print(s)
    def load_tiles_db(tilespath):
        tiles = Path(tilespath)
        if not tiles.is_dir():
            # assume pickle file
            vlog(f'Loading pickled database file: {tilespath}.')
            with lzma.open(tiles) as fp:
                db = pickle.load(fp)
            if not isinstance(db, dict):
                vlog(f'Loaded pickle file {args.tiles} is not a valid tiles database.')
                return None
            else:
                return db
        else:
            db = {}
            def processTilesJson(tilesJson):
                for feat in tilesJson['features']:
                    imgid = feat['properties']['id']
                    seqid = feat['properties']['sequence_id']
                    angle = feat['properties']['compass_angle']
                    coord = feat['geometry']['coordinates']
                    db[imgid] = {
                        'seqid': seqid,
                        'angle': angle,
                        'lat': coord[1],
                        'lon': coord[0],
                        'is_pano': feat['properties']['is_pano']
                    }
                    #vlog(json.dumps(db[imgid]))

            for tilefile in tiles.glob('mly1_public*'):
                with tilefile.open() as fp:
                    tilesJson = json.load(fp)
                    processTilesJson(tilesJson)
            return db

    if args.tiles is not None:
        db = load_tiles_db(args.tiles)
    else:
        db = None
    sqloutdir = None
    if args.sqloutdir is not None:
        if db is None:
            print(f'Cannot output SQL files without tiles information.')
            sys.exit(1)
        sqloutdir = Path(args.sqloutdir)
        os.makedirs(sqloutdir, exist_ok=True)
        if not sqloutdir.is_dir():
            print(f'Failed to make SQL output directory: {sqloutdir}')
            sys.exit(1)

    def sqlout(stem, seqid, typ, s):
        if sqloutdir is None: return
        outdir = sqloutdir / Path(str(seqid))
        os.makedirs(outdir, exist_ok=True)
        outfilename = outdir / Path(f'{str(stem)}_{typ}').with_suffix('.sql')
        if args.overwrite or not outfilename.exists():
            with open(outfilename,'w') as fp: fp.write(s)
        else:
            vlog(f'WARNING: file {outfilename} already exists and overwriting is not enabled!')

    def do_file(filename):
        if args.log:
            outfile = Path(filename).with_suffix('.out')
            logfp = open(outfile, 'w')
        def vlog(s):
            if args.log:
                logfp.write(f'{s}\n')
            if args.verbose:
                print(s)

        vlog(f'Loading "{filename}".')
        if Path(filename).suffix == '.npz':
            with np.load(filename) as f:
                predict = f['predict']
                modelname = str(f['modelname'])
        else:
            predict = np.load(filename)
            modelname = None
        origstem = Path(filename).stem
        try:
            imgid = int(origstem)
        except:
            imgid = origstem
        vlog(f'Assuming imgid={imgid}')

        jpgfile = Path(filename).with_suffix('.jpg')
        rgbimg = None
        if jpgfile.exists() and not args.centres_only:
            img = cv2.imread(str(jpgfile))
            rgbimg = img[:, :, ::-1]
            if not args.fast:
                vlog(f'Simple brightness: {simple_brightness(img)}')
                vlog(f'Finley brightness: {finley_brightness(img)}')
                vlog(f'Michaelson contrast: {michaelson_contrast(img)}')
                vlog(f'RMS contrast: {rms_contrast(img)}')
                vlog(f'Is low contrast?: {is_low_contrast(rgbimg, fraction_threshold=0.35)}')
                vlog(f'Laplacian: {laplacian(img)}')
            vlog(f'Skimage contrast: {skimage_contrast(rgbimg)}')
            vlog(f'Tone-mapping score: {compute_hdr(img)}')

        vlog(f'Matrix shape: {predict.shape}.')
        if (imgid in db and db[imgid]['is_pano']) or predict.shape[1] >= predict.shape[0] * 2:
            # panoramic
            predictplus = np.append(predict, predict[:,:predict.shape[1]//4],axis=1)
            vlog(f'Assuming panoramic input, extending width to {predictplus.shape[1]}.')
            is_pano = True
        else:
            predictplus = predict
            is_pano = False

        distance=args.road_peaks_distance
        if distance is None:
            distance = int(2000 * predict.shape[1] // 5760)

        prominence=args.road_peaks_prominence
        if prominence is None:
            prominence = int(100 * predict.shape[0] // 2880)

        vlog(f'Seeking road centres (using pixel segmentation; distance={distance}, prominence={prominence})...')

        centres=road_centres(predictplus, distance=distance, prominence=prominence)
        vlog(f'Found road centres: {centres}.')
        dataset = args.dataset or 'citys'
        if args.centres_only:
            if args.log:
                logfp.close()
            return

        if modelname is not None and args.dataset is None:
            dataset = modelname.split('_')[-1] 

        def sql(imgid, stem, angle_delta=0.0):
            if imgid not in db:
                vlog(f'Image ID {imgid} not found in tiles database, skipping SQL output.')
                return
            entry = db[imgid]
            seq_id = entry['seqid']
            lon = entry['lon']
            lat = entry['lat']
            angle = entry['angle'] + angle_delta
            geo=f'ST_SetSRID(ST_MakePoint({lon}, {lat}),4326)::geometry(POINT, 4326)'
            cityname=args.cityname
            url=f'{args.urlprefix}/{cityname}/{seq_id}/{stem}.jpg'
            syspath=f'{args.dirprefix}/{cityname}/{seq_id}/{stem}.jpg'
            sqlout(stem, seq_id, 'insert', f"""
WITH image_ins AS (
INSERT INTO image (url, system_path, cityname, enabled) VALUES ('{url}', '{syspath}', '{cityname}', false) ON CONFLICT DO NOTHING RETURNING image_id
)
INSERT INTO image_geo (angle_deg, geo, image_id) SELECT {angle} AS angle_deg, {geo} AS geo, image_id FROM image_ins ON CONFLICT DO NOTHING;
""")
            sqlout(stem, seq_id, 'enable', f"UPDATE image SET enabled=true WHERE system_path = '{syspath}';\n")

        if jpgfile.exists() and is_pano:
            image = Image.open(jpgfile)
            origwidth = image.size[0]
            subimages, subsegms, infos = crop_panoramic_image(image, torch.from_numpy(predict), centres)
            #vlog(f'crop_panoramic_image: info={json.dumps(infos, default=int, indent=2)}')
            if args.cropsdir is not None:
                os.makedirs(args.cropsdir, exist_ok=True)
            for (subimg, info) in zip(subimages, infos):
                #vlog(f'imgx={info["imgx"]}')
                imgx = info['imgx']
                subimgfilename = Path(filename).with_stem(f'{origstem}_x{imgx}').with_suffix('.jpg')
                stem = subimgfilename.stem
                if args.cropsdir is not None:
                    subimgfilename = Path(args.cropsdir) / subimgfilename.name
                if args.overwrite or not subimgfilename.exists():
                    vlog(f'Cropped image file: {subimgfilename}')
                    subimg.save(subimgfilename)
                else:
                    vlog(f'Cropped image file (already exists): {subimgfilename}')
                sql(imgid, stem, 360.0 * imgx / origwidth if origwidth != 0.0 else 0.0)
        else:
            # non-pano, or no jpg file found
            sql(imgid, origstem)

        if args.fast:
            if args.log:
                logfp.close()
            return

        colors = [ '#000000' ]
        if args.palette_file is not None:
            with open(args.palette_file) as fp:
                colors = []
                for c in fp:
                    if len(c.strip()) > 0: colors.append(c.strip())

        vlog(f'Generating mask image.')
        blank = torch.zeros(3, predictplus.shape[0], predictplus.shape[1], dtype=torch.uint8)
        rgbimgplus = None
        if rgbimg is not None and is_pano:
            # Simulate augmentation of underlying image in the same way that was
            # done for the semantic segmentation process
            rgbimgplus = np.append(rgbimg, rgbimg[:,:rgbimg.shape[1]//4],axis=1)
            if (rgbimg.shape[0]*3//4) % predictplus.shape[0] == 0:
                # crop bottom fourth of image, if it results in an integer scaledown_factor
                rgbimgplus = rgbimgplus[:rgbimg.shape[0]*3//4,:]
            #scaledown_factor = rgbimgplus.shape[0] // predictplus.shape[0]
            # Thus, ensure that rgbimgplus has the same shape as predictplus
            rgbimgplus = cv2.resize(rgbimgplus, (predictplus.shape[1], predictplus.shape[0]))
            # Rearrange dimensions to make Torch draw_segmentation_masks happy
            seginput = torch.from_numpy(np.transpose(rgbimgplus, (2, 0, 1)))
            alpha = args.mask_alpha
        elif rgbimg is not None:
            # Rearrange dimensions to make Torch draw_segmentation_masks happy
            rgbimgresized = cv2.resize(rgbimg, (predictplus.shape[1], predictplus.shape[0]))
            seginput = torch.from_numpy(np.transpose(rgbimgresized, (2, 0, 1)))
            alpha = args.mask_alpha
        else:
            # Can't find base image; so assume blank
            seginput = blank
            alpha = 1

        # Split the predictplus array of integers (ranging 0..N) into N separate arrays
        # of Boolean values, because this is what draw_segmentation_masks needs.
        masks_bool = np.array([predictplus == x for x in np.unique(predictplus)])

        # If there are insufficient colours then add grey to the list as many times as needed
        while len(colors) < masks_bool.shape[0]:
           colors.append('#808080')

        # maskedimg has the combined masks drawn on top of the underlying image
        maskedimg = draw_segmentation_masks(seginput, torch.tensor(masks_bool,dtype=torch.bool), alpha=alpha, colors=colors).numpy()
        # transpose it back to OpenCV style
        maskedimg = np.transpose(maskedimg, [1, 2, 0])

        # This combined mask is drawn on top of blank underlying image, for further analysis
        mask = draw_segmentation_masks(blank, torch.tensor(masks_bool,dtype=torch.bool)).numpy()
        # transpose it back to OpenCV style
        mask = np.transpose(mask, [1, 2, 0])

        # Further analysis...
        gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        if args.blur or args.blurfile:
            k=5
            vlog(f'Running Gaussian blur with kernel {k}x{k}.')
            blur = cv2.GaussianBlur(gray, (k, k), 1)
            if args.blurfile:
                vlog(f'Writing blur image "{args.blurfile}".')
                cv2.imwrite(args.blurfile,blur)
        else:
            blur = None

        edgeImg = cv2.Canny(blur if blur is not None else gray, 40, 255)
        if args.edgefile:
            vlog(f'Writing edges image to "{args.edgefile}".')
            cv2.imwrite(args.edgefile, edgeImg)

        #lines = cv2.HoughLinesP(edgeImg, 1, np.pi / 180, 50, None, 50, 10)
        rho = float(args.houghlines_rho or 1)
        theta = float(args.houghlines_theta or np.pi/120)
        threshold = int(args.houghlines_threshold or 120)
        min_theta = float(args.houghlines_min_theta or np.pi/36)
        max_theta = float(args.houghlines_max_theta or np.pi-np.pi/36)
        vlog(f'Running HoughLines (rho={rho}, theta={theta}, threshold={threshold}, min_theta={min_theta}, max_theta={max_theta}).')
        lines = cv2.HoughLines(edgeImg, rho, theta, threshold, min_theta=min_theta, max_theta=max_theta)

        cdst = cv2.cvtColor(edgeImg, cv2.COLOR_GRAY2BGR)

        # https://stackoverflow.com/questions/57535865/extract-vanishing-point-from-lines-with-open-cv
        if lines is not None:
            vlog(f'Line count: {len(lines)}')
            for line in lines:
                rho,theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 10000*(-b))
                y1 = int(y0 + 10000*(a))
                x2 = int(x0 - 10000*(-b))
                y2 = int(y0 - 10000*(a))
                cv2.line(cdst,(x1,y1),(x2,y2),(0,255,0),1, cv2.LINE_AA)

        if args.linefile:
            vlog(f'Writing lines image to "{args.linefile}".')
            cv2.imwrite(args.linefile, cdst)

        blobs = np.copy(cdst)

        kernel = np.ones((3,3),np.uint8)
        blobs = cv2.erode(blobs,kernel,iterations=1)
        kernel = np.ones((9,9),np.uint8)
        blobs = cv2.dilate(blobs,kernel,iterations=1)
        kernel = np.ones((11,11),np.uint8)
        blobs = cv2.erode(blobs,kernel,iterations=1)
        blobs = cv2.dilate(blobs,kernel,iterations=1)
        if args.blobfile:
            vlog(f'Writing blobs image to "{args.blobfile}".')
            cv2.imwrite(args.blobfile, blobs)

        grayblobs = cv2.cvtColor(blobs,cv2.COLOR_BGR2GRAY)
        grayblobs1d = np.count_nonzero(grayblobs,axis=0)
        #blobvp1 = grayblobs1d.argmax()
        #blobvp2 = (blobvp1 + predict.shape[1]//2) % predict.shape[1]
        #print(f'blobvp1={blobvp1} blobvp2={blobvp2}')

        blobvps = find_peaks(grayblobs1d, distance=predict.shape[1]//4)[0]
        vlog(f'Found vanishing points (using coalesced blobs): {blobvps}.')

        sqrw = 30
        sqry = 5

        maskrgb = maskedimg
        img = Image.fromarray(maskrgb)
        draw = ImageDraw.Draw(img)
        if not args.no_houghtransform_road_centrelines:
            for b1 in blobvps:
                color = (256,0,0)
                draw.line((b1,0,b1,mask.shape[1]),width=6,fill=color)
                #b2 = (b1 + predict.shape[1]//2) % predict.shape[1]
                #draw.line((b2,0,b2,mask.shape[1]),width=1,fill=(128,0,0))
                draw.rectangle((b1 - sqrw//2, sqry, b1 + sqrw//2, sqry+sqrw), outline=color, fill=color)
        if not args.no_segmentation_road_centrelines:
            for c1 in centres:
                color = (0,256,0)
                draw.line((c1,0,c1,mask.shape[1]),width=6,fill=color)
                draw.ellipse((c1 - sqrw//2, sqry, c1 + sqrw//2, sqry+sqrw), outline=color, fill=color)
                #c2 = (c1 + predict.shape[1]//2) % predict.shape[1]
                #draw.line((c2,0,c2,mask.shape[1]),width=1,fill=(0,256,0))
                ####
                #blobDistThreshold = 100
                #if abs(blobvp1 - c1) < blobDistThreshold or abs(blobvp2 - c1) < blobDistThreshold or \
                #  abs(blobvp1 - c2) < blobDistThreshold or abs(blobvp2 - c2) < blobDistThreshold:
                #    print(f'c1={c1} c2={c2}')
                #    draw.line((c2,0,c2,mask.size[1]),fill=128)
                #    break

        if args.maskfile:
            if type(args.maskfile) != str:
                args.maskfile = Path(filename).with_stem(f'{origstem}_mask').with_suffix('.jpg')
            vlog(f'Writing mask image to "{args.maskfile}".')
            img.save(args.maskfile)

        if args.plusfile and rgbimgplus is not None:
            vlog(f'Writing "plus" image to "{args.plusfile}".')
            Image.fromarray(rgbimgplus).save(args.plusfile)

        if args.overfile and rgbimg is not None:
            if rgbimgplus is not None:
                # rgbimgplus should be original JPG in RGB format, as a numpy array, with augmentation
                # (the leftmost 25% of the image is appended onto the right side; this matches the
                # transformation carried out prior to semantic segmentation for panoramic images)
                # Hence, the image coordinates match the matrix coordinates (predictplus) already.
                orig_as_pil_rgb = Image.fromarray(rgbimgplus)
                ratio = 1
            else:
                # rgbimg should be original JPG in RGB format, as a numpy array
                imgw = rgbimg.shape[1]
                matw = predict.shape[1]
                # Matrix coordinates are likely to be scaled down compared to image coords.
                ratio = imgw // matw
                sqrw *= ratio
                sqry *= ratio
                orig_as_pil_rgb = Image.fromarray(rgbimg)
            draw = ImageDraw.Draw(orig_as_pil_rgb)
            if not args.no_houghtransform_road_centrelines:
                for b1 in blobvps:
                    b1 *= ratio
                    color = (256,0,0)
                    draw.line((b1,0,b1,rgbimg.shape[0]),width=6,fill=color)
                    draw.rectangle((b1 - sqrw//2, sqry, b1 + sqrw//2, sqry+sqrw), outline=color, fill=color)
            if not args.no_segmentation_road_centrelines:
                for c1 in centres:
                    c1 *= ratio
                    color = (0,256,0)
                    draw.line((c1,0,c1,rgbimg.shape[0]),width=6,fill=color)
                    draw.ellipse((c1 - sqrw//2, sqry, c1 + sqrw//2, sqry+sqrw), outline=color, fill=color)

            if type(args.overfile) != str:
                args.overfile = Path(filename).with_stem(f'{origstem}_over').with_suffix('.jpg')
            vlog(f'Writing centrelines over original image to "{args.overfile}".')
            orig_as_pil_rgb.save(args.overfile)

        if args.log:
            logfp.close()

    np_extensions = ['npz', 'npy']
    if args.filelist:
        with open(args.filename) as fp:
            for name in fp:
                p = Path(name.strip())
                if p.is_file() and p.suffix.lower()[1:] in np_extensions:
                    do_file(p)
    else:
        do_file(args.filename)

if __name__=='__main__':
    main()

# vim: ai sw=4 sts=4 ts=4 et
