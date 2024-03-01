# Volunteered street view imagery processing and filtering for the human perception survey  

Tools to help filter 'volunteered street view imagery' (VSVI) such as photos found on Mapillary and similar sources. Part of the [human perception survey project](https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/percept).

# Set-up

Have a recent version of Python3 and with pip: `pip install -r requirements.txt`.

# Alternative: Docker Set-up

Instead of installing Python3 and the necessary packages on your system, you can use [Docker](https://www.docker.com) containers. A Docker image with a suitable environment can be built (one time) by running the script `./build.sh` and then the other commands in the suite can be run as shown in the sections below but first prepending `./run.sh` in front of them. For example, 
  - `./run.sh ./torch_segm_images.py --gpu 0 -v image1.jpg`

# Commands

## `mapillary_jpg_download.py`

Script to download all mapillary street view imagery within a given latitude/longitude boundary 'box'.

This script is provided as-is. The usage of this script, compliance with Mapillary licencing and acceptable use terms, as well as any Internet service provider terms, is entirely your responsibility.

### Configuration

This program can be run entirely from the command-line, or it can be configured using a token file and/or a configuration JSON file.

The following information must be provided in one form or another:
* Mapillary API access token
	- (from Developer section of Mapillary, after you Register an Application, copy the 'Client Token' field)
* Tile cache directory
* Image sequence download directory
* Geographic bounding box of imagery to download: west (longitude), south (latitude), east (longitude), north (latitude)

The command-line reference can be found below.

#### Token file

The API token is a secret that should not be shared in code repositories or any
publicly-accessible archives. It is probably not a good idea to normally
provide the token on the command-line because most command shells will save
your command history in a file, so you may want to clear that history (e.g. in
bash use the command `history -c`) if you do use the `--token` argument.

If you choose to put your API token into a file then please put it entirely by
itself into a single, simple text file. You can specify the `--token-file`
argument on the command-line to feed it to the program, or simply use the
default name `token.txt` and the program will find it in the current directory.

Do not commit the token file to a repository (e.g. you may want to add the
filename to your `.gitignore` if you are using git).

#### Configuration file

Tile cache directory, image sequence directory and geographic bounding box can
be provided in a JSON file that should look like this:

    {
            "bounding_box": {
                    "west": 4.7149,
                    "south": 52.2818,
                    "east": 5.1220,
                    "north": 52.4284
            },
            "tile_cache_dir": "<tile directory>",
            "seqdir": "<sequence directory>"
    }

An example may be found in `examples/greater-amsterdam.json`.

The `-c` command-line argument can be used to feed the configuration file to
the program.

`tile_cache_dir` and `seqdir` are important because they are the directores to which this script will download the Mapillary tiles GeoJSON files and the actual street view imagery sequences, respectively.

### Examples

* Assuming your API token is saved in `token.txt`:
  - `./mapillary_jpg_download.py -c examples/greater-amsterdam.json`
* Assuming your API token is saved in `mytoken.txt`:
  - `./mapillary_jpg_download.py -c examples/greater-amsterdam.json --token-file mytoken.txt`
* Fully command-line:
  - `./mapillary_jpg_download.py --token 'MLY...' --tile-cache-dir tiles --seqdir seqs --west 4.7 --south 52.2 --east 5.12 --north 52.4`
* Reduce number of retries to 6, will stop running if free disk space falls below 50GB, and will store failed-to-download image IDs in a file:
  - `./mapillary_jpg_download.py -c examples/greater-amsterdam.json --num-retries 6 --required-disk-space 50 --failed-imgid-file list-of-failed-imgids.txt`

### Usage

    mapillary_jpg_download.py [--configfile FILENAME] [options]

    options:
      -h, --help               show this help message and exit
      --configfile FILENAME, --config FILENAME, -c FILENAME
                               Configuration file to process
      --quiet, -q              Run in quiet mode
      --overwrite, -O          Overwrite any existing output file
      --tile-cache-dir DIR     Directory in which to store the Mapillary GeoJSON tiles cache
      --tile-list-file FILE    Work on the listed tiles only, identified by tile cache filename, 1 per line
      --imgid-file FILE        Only download the Mapillary image IDs found in this file (1 ID listed per line)
      --failed-imgid-file      Record failed-to-download Mapillary image IDs into this file (for later use with --imgid-file)
      --seqdir DIR             Directory in which to store street view imagery sequences (a large amount of image data)
      --token TOKEN            Mapillary API token (see Developers help for Mapillary)
      --token-file FILE        Alternatively, read the token from this file (with the token written on a single line)
      --required-disk-space    Will stop run less than this number in gigabytes is available. (default: 100)
      --num-retries NUM        Number of times to retry if there is a network failure. (default: 8)
      --west LON               Western boundary (longitude)
      --south LAT              Southern boundary (latitude)
      --east LON               Eastern boundary (longitude)
      --north LAT              Northern boundary (latitude)

## `make_tiles_db.py`

Take the GeoJSON tiles database (obtained from Mapillary) and condense it into a pickled database. Not strictly necessary, but makes further processing commands much faster on large imagery collections.

### Examples

* Produce a pickled database file named `my-tiles-database.pkl` from the directory `my-tiles-directory/`:
  - `./make_tiles_db.py -o my-tiles-database.pkl my-tiles-directory/`

### Usage

    make_tiles_db.py -o FILENAME DIRECTORY

    required:
      DIRECTORY                             Directory containing Mapillary GeoJSON tiles cache
      -o FILENAME, --output FILENAME        Write pickled database into FILENAME
      

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

