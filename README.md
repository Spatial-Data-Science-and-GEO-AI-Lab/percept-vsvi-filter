# Volunteered street view imagery processing and filtering for the human perception survey  

Tools to help filter 'volunteered street view imagery' (VSVI) such as photos found on Mapillary and similar sources. Part of the [human perception survey project](https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept).

# Set-up

Have a recent version of Python3 and with pip: `pip install -r requirements.txt`.

# Alternative: Docker Set-up

Instead of installing Python3 and the necessary packages on your system, you can use [Docker](https://www.docker.com) containers. A Docker image with a suitable environment can be built (one time) by running the script `./build.sh` and then the other commands in the suite can be run as shown in the sections below but first prepending `./run.sh` in front of them. For example, 
  - `./run.sh ./torch_segm_images.py --gpu 0 -v image1.jpg`

# Commands

## `make_tiles_db.py`

Take the GeoJSON tiles database (obtained from Mapillary) and condense it into a pickled database. Not strictly necessary, but makes further processing commands much faster on large imagery collections.

### Examples

* Produce a pickled database file named `my-tiles-database.pkl` from the directory `my-tiles-directory/`:
  - `./make_tiles_db.py -o my-tiles-database.pkl my-tiles-directory/`

## `torch_segm_images.py`

Run semantic segmentation on images using PyTorch and output the result into accompanying '.npz' files for further processing.

### Examples

* Run with the first GPU on `image1.jpg`, in verbose mode:
  - `./torch_segm_images.py --gpu 0 -v image1.jpg`

* Run on `image1.jpg` and recursively on directory `dir_of_jpgs`:
  - `./torch_segm_images.py -v -r image1.jpg dir_of_jpgs/`

* Run recursively on `dir_of_pngs` and `dir_of_jpgs` looking for PNG and JPG files:
  - `./torch_segm_images.py -v -e png jpeg jpg -r dir_of_pngs dir_of_jpgs/`

### Usage

    torch_segm_images.py [options] PATH [PATHS]

    positional arguments:
      PATH                  Filenames or directories to process as input (either images or filelists, see -e and -F)

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Run in verbose mode
      --filelist, -F        Supplied paths are actually a list of image filenames, one per line, to process (does not work with -r)
      --output-filelist FILE
                            Record the names of saved numpy output files in this given FILE.
      --output-extension EXT
                            Output filename extension (default: npz)
      --recursive, -r       Recursively search for images in the given directory and subdirectories (only if -F not enabled).
      --image-extensions EXT [EXT ...], -e EXT [EXT ...]
                            Image filename extensions to consider (default: jpg jpeg). Case-insensitive.
      --no-detect-panoramic
                            Do not try to detect and correct panoramic images
      --scaledown-factor SCALEDOWN_FACTOR, -s SCALEDOWN_FACTOR
                            Image scaling down factor
      --scaledown-interp SCALEDOWN_INTERP
                            Interpolation method (NEAREST (0), LANCZOS (1), BILINEAR (2), BICUBIC (3), BOX (4) or HAMMING (5)).
      --overwrite, -O       Overwrite any existing output file
      --dry-run             Do not actually write any output file
      --modelname MODEL     Use a specified model (from gluoncv.model_zoo)
      --gpu [N], -G [N]     Use GPU (optionally specify which one)
      --exclusion-pattern REGEX, -E REGEX
                            Regex to indicate which files should be excluded from processing.

## `torch_process_segm.py`

Process the segmentation files produced by `torch_segm_images.py`, calculate road centers and various image quality metrics, and output SQL statements to populate the human perception survey database.

### Examples

* Verbosely process the generated NPZ files (listed in `list-of-npz-files.txt`), using a tiles database found in `my-tiles-database.pkl`, output `.out` files and cropped JPGs to the same directory as each NPZ file, put generated SQL into the `sqldir/` directory, use `/system/path/to/images` for the `system_path` in the generated SQL and `/web/path/to/img/folder` as the base URL for the images in the generated SQL, and MyCityName as the city name.
  - `./torch_process_segm.py -v --log -T my-tiles-database.pkl -D /system/path/to/images -U /web/path/to/img/folder -C MyCityName -S sqldir/ -F list-of-npz-files.txt`

### Usage

    torch_process_segm.py [options] FILENAME

    positional arguments:
      FILENAME              Saved numpy (.npz or .npy) file to process, or list of such files (see -F)

    options:
      -h, --help            show this help message and exit
      --verbose, -v         Run in verbose mode
      --filelist, -F        Supplied path is actually a list of numpy filenames, one per line, to process.
      --fast                Fast mode, skip most functionality except: Road Finding, SKImage Contrast, Tone-mapping, and panoramic-image cropping.
      --centres-only        Skip all functionality except Road Finding
      --overwrite, -O       Overwrite output files
      --sqloutdir DIR, -S DIR
                            Directory to output SQL files
      --cropsdir DIR        Directory to output cropped JPGs (default: same directory as original image)
      --tiles FILENAME-OR-DIR, -T FILENAME-OR-DIR
                            Directory to find tiles files in JSON, or a tiles picklefile
      --dirprefix DIRPREFIX, -D DIRPREFIX
                            prefix of system path for images
      --urlprefix URLPREFIX, -U URLPREFIX
                            prefix of URL for images
      --cityname CITYNAME, -C CITYNAME
                            name of city associated with the given numpy files
      --log                 Save verbose output to .out file
      --blur                Run Gaussian blur before finding edges
      --palette-file FILENAME, -P FILENAME
                            File with list of colour names for mask output, one per line
      --mask-alpha ALPHA    Alpha transparency value when drawing mask over image (0 = fully transparent; 1 = fully opaque)
      --no-houghtransform-road-centrelines
                            Do not draw Hough transform-based road centrelines
      --no-segmentation-road-centrelines
                            Do not draw segmentation-based road centrelines
      --maskfile [FILENAME], -m [FILENAME]
                            Output filename for mask image
      --plusfile FILENAME, -p FILENAME
                            Output filename for "plus" image; with the leftmost 25-percent appended to the righthand side
      --overfile [FILENAME], -o [FILENAME]
                            Output filename for image with centrelines drawn over
      --edgefile FILENAME, -e FILENAME
                            Output filename for edges image
      --linefile FILENAME, -l FILENAME
                            Output filename for lines image
      --blurfile FILENAME, -B FILENAME
                            Output filename for Gaussian blurred image
      --blobfile FILENAME, -b FILENAME
                            Output filename for blobs image
      --dataset DATASET     Override segmentation dataset name (for visualisation)
      --road-peaks-distance N
                            Distance between peaks of road pixels
      --road-peaks-prominence N
                            Prominence of peaks of road pixels

## `filter_output.py`

Applies filtering criteria to a list of `.out` files (produced by `torch_process_segm.py`) in order to determine which images to accept or reject. Writes a list of accepted image IDs (one per line) to the output file.

### Example:

* Get a list of out-files and then run them through the filter
  - `find my-dir -name '*.out' > list-of-out-files.txt`
  - `./filter_output.py -v -o list-of-accepted-imageids.txt -F list-of-out-files.txt`

### Usage:

    filter_output.py [options] FILENAME
      FILENAME              Saved out file to process, or list of such files (see -F)

    options:
      -h, --help            show this help message and exit
      --filelist, -F        Supplied path is actually a list of numpy filenames, one per line, to process.
      --output FILENAME, -o FILENAME
                            File to write the list of filtered image identifiers
      --verbose, -v         Run in verbose mode
      --contrast-threshold NUMBER, -C NUMBER
                            Minimum contrast (0.35 default)
      --tone-mapping-threshold NUMBER, -H NUMBER
                            Minimum tone mapping score (0.35 default)
      --tone-mapping-floor NUMBER    Tone mapping floor (0.8 default)

