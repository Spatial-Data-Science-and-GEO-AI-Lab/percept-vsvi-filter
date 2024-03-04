#!/usr/bin/env python3
# Simple program to load an image, run a semantic segmentation model on it
# (from Torch) and save the results in a compressed numpy file.
#
# Torch-using code re-adapted from the port of this repository's code by Ilse Abril Vázquez Sánchez:
# https://github.com/Spatial-Data-Science-and-GEO-AI-Lab/StreetView-NatureVisibility/
#
# See README.md.
# Licence:
#   Copyright 2023 Matthew Danish
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
import argparse
from time import time
import numpy as np
from pathlib import Path
import sys
import re
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image, ImageFile

parser = argparse.ArgumentParser(prog='torch_segm_images.py', description='Run semantic segmentation using a Mask2Former model from HuggingFace (see https://huggingface.co/models?search=mask2former)')
parser.add_argument('paths', metavar='PATH', nargs='+', help='Filenames or directories to process as input (either images or filelists, see -e and -F)')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Run in verbose mode')
parser.add_argument('--filelist', '-F', action='store_true', default=False, help='Supplied paths are actually a list of image filenames, one per line, to process (does not work with -r)')
parser.add_argument('--output-filelist', metavar='FILE', default=None, help='Record the names of saved numpy output files in this given FILE.')
parser.add_argument('--output-extension', metavar='EXT', default='npz', help='Output filename extension (default: npz)')
parser.add_argument('--recursive', '-r', default=False, action='store_true',help='Recursively search for images in the given directory and subdirectories (only if -F not enabled).')
parser.add_argument('--image-extensions', '-e', metavar='EXT', nargs='+', default=['jpg', 'jpeg'], help='Image filename extensions to consider (default: jpg jpeg). Case-insensitive.')
parser.add_argument('--no-detect-panoramic', default=False, action='store_true',help='Do not try to detect and correct panoramic images')
parser.add_argument('--scaledown-factor', '-s', default=4, type=float, help='Image scaling down factor')
parser.add_argument('--scaledown-interp', default=3, type=int, help='Interpolation method (NEAREST (0), LANCZOS (1), BILINEAR (2), BICUBIC (3), BOX (4) or HAMMING (5)).')
parser.add_argument('--overwrite', '-O', action='store_true', default=False, help='Overwrite any existing output file')
parser.add_argument('--dry-run', action='store_true', default=False, help='Do not actually write any output file')
parser.add_argument('--modelname', metavar='MODEL', help='Use a specified model (from https://huggingface.co/models?search=mask2former)',default="facebook/mask2former-swin-large-cityscapes-semantic")
parser.add_argument('--gpu', '-G', metavar='N', nargs='?', default=None, const=True, help='Use GPU (optionally specify which one)')
parser.add_argument('--exclusion-pattern', '-E', metavar='REGEX', default='.*(npz|mask|out|_x[0-9]+).*', help='Regex to indicate which files should be excluded from processing.')

def main():
    args = parser.parse_args()
    def vlog(s):
        if args.verbose:
            print(s)

    if args.gpu is not None:
        vlog(f'Using GPU ({args.gpu}).')
        if type(args.gpu)=='str':
            device = torch.device('cude' if torch.cuda.is_available() else 'cpu', int(args.gpu))
        elif type(args.gpu)=='int':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', args.gpu)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        vlog('Using CPU.')
        device = torch.device('cpu')

    vlog(f'device={device}')
    vlog(f'Loading model "{args.modelname}".')
    processor = AutoImageProcessor.from_pretrained(args.modelname)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.modelname)
    model = model.to(device)

    def do_file(inputpath):
        try:
            inputfilename = inputpath.name
            outputpath = inputpath.with_suffix(f'.{args.output_extension}')
            if outputpath.exists() and not args.overwrite:
                try:
                    with np.load(outputpath) as f:
                        if 'predict' in f:
                            vlog(f'Skipping existing output file "{outputpath}".')
                            if args.output_filelist is not None:
                                with open(args.output_filelist, 'a') as fp:
                                    fp.write(f'{str(outputpath)}\n')
                            return
                except:
                    pass
            vlog(f'Loading image "{inputpath}"...')
            t1 = time()
            img = Image.open(inputpath)

            vlog(f'Image size={img.size[0]}x{img.size[1]}.')
            if not args.no_detect_panoramic and img.size[0] >= img.size[1]*2:
                img = img.crop((0, 0, img.size[0], img.size[1]*3//4))
                vlog(f'Assuming panoramic image, cropping to {img.size[0]}x{img.size[1]}.')

            if args.scaledown_factor != 1:
                img = img.resize(( int(img.size[0]//args.scaledown_factor),
                                   int(img.size[1]//args.scaledown_factor) ),
                                 resample=args.scaledown_interp)
                vlog(f'Scaling down image to {img.size[0]}x{img.size[1]}.')

            t2 = time()
            vlog(f'Loading complete. Runtime: {(t2-t1):.2f}s')

            vlog('Running model...')
            t1 = time()

            # Preprocess the image using the image processor
            inputs = processor(images=img, return_tensors="pt")

            # Perform a forward pass through the model to obtain the segmentation
            with torch.no_grad():
                # Check if a GPU is available
                if args.gpu is not None and torch.cuda.is_available():
                    # Move the inputs to the GPU
                    inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    # Perform the forward pass through the model
                    outputs = model(**inputs)
                    # Post-process the semantic segmentation outputs using the processor and move the results to CPU
                    segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0].to('cpu')
                else:
                    # Perform the forward pass through the model
                    outputs = model(**inputs)
                    # Post-process the semantic segmentation outputs using the processor
                    segmentation = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]

            predict = segmentation.numpy()

            t2 = time()
            vlog(f'Complete. Runtime: {(t2-t1):.2f}s.')

            if not args.dry_run:
                vlog(f'Saving predictions (shape={predict.shape}) into "{outputpath}".')
                np.savez_compressed(str(outputpath), predict=predict, modelname=args.modelname)
                if args.output_filelist is not None:
                    with open(args.output_filelist, 'a') as fp:
                        fp.write(f'{str(outputpath)}\n')
        except Exception as e:
            vlog(f'Failed: {e}. Skipping.')
            errpath = inputpath.with_suffix(f'.err')
            with open(errpath, 'w') as fp:
                fp.write(str(e) + '\n')

    image_extensions = [ e.lower() for e in args.image_extensions ]
    exclude = re.compile(args.exclusion_pattern)
    if args.filelist:
        for filelist in args.paths:
            with open(filelist) as fp:
                for name in fp:
                    p = Path(name.strip())
                    if p.is_file() and p.suffix.lower()[1:] in image_extensions:
                        if not exclude.match(p.name):
                            do_file(p)
    else:
        def recur(paths):
            for name in paths:
                p = Path(name)
                if p.is_file() and p.suffix.lower()[1:] in image_extensions:
                    if not exclude.match(p.name):
                        do_file(p)
                elif args.recursive and p.is_dir():
                    recur(p.iterdir())
        recur(args.paths)

if __name__=='__main__':
    main()

# vim: ai sw=4 sts=4 ts=4 et
