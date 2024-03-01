#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import os

parser = argparse.ArgumentParser(prog='filter_output.py', description='Filter output files based on criteria')
parser.add_argument('filename', metavar='FILENAME', help='Saved out file to process, or list of such files (see -F)')
parser.add_argument('--filelist', '-F', action='store_true', default=False, help='Supplied path is actually a list of numpy filenames, one per line, to process.')
parser.add_argument('--output', '-o', metavar='FILENAME', required=True, default=None, help='File to write the list of filtered image identifiers')
parser.add_argument('--verbose', '-v', action='store_true', default=False, help='Run in verbose mode')
parser.add_argument('--contrast-threshold', '-C', metavar='NUMBER', default=0.35, type=float, help='Minimum contrast')
parser.add_argument('--tone-mapping-threshold', '-H', metavar='NUMBER', default=0.35, type=float, help='Minimum contrast')
parser.add_argument('--tone-mapping-floor', metavar='NUMBER', default=0.8, type=float, help='Tone mapping floor')

out_extensions = ['out']

contrast_tag = 'Skimage contrast: '
tone_mapping_tag = 'Tone-mapping score: '
centres_tag = 'Found road centres: ['
imgid_tag = 'Assuming imgid='

def main():
    args = parser.parse_args()
    def vlog(s):
        if args.verbose:
            print(s)

    def do_file(fname, outfp):
        contrast = None
        tone_mapping = None
        imgid = None
        centres = None
        with open(fname) as fp:
            for line in fp:
                if contrast_tag in line:
                    contrast = float(line[len(contrast_tag):])
                elif tone_mapping_tag in line:
                    tone_mapping = float(line[len(tone_mapping_tag):])
                elif imgid_tag in line:
                    imgid = int(line[len(imgid_tag):])
                elif centres_tag in line:
                    closebracketpos = line.rfind(']')
                    if closebracketpos == -1:
                        vlog(f'Invalid road centres line: {line}')
                    else:
                        centres = list(map(int,filter(lambda x: x, line[len(centres_tag):closebracketpos].strip().split(' '))))
                    
        v = contrast + max(0, tone_mapping - args.tone_mapping_floor)
        accept = v > args.contrast_threshold and len(centres) == 1
        vlog(f'imgid={imgid} contrast={contrast} Tone-mapping={tms} road centres={centres} v={v} accept?: {accept}')
        if accept: outfp.write(f'{imgid}\n')

    with open(args.output, 'w') as outfp:
        if args.filelist:
            with open(args.filename) as fp:
                for name in fp:
                    p = Path(name.strip())
                    if p.is_file() and p.suffix.lower()[1:] in out_extensions:
                        do_file(p, outfp)
        else:
            do_file(args.filename, outfp)
if __name__=='__main__':
    main()

# vim: ai sw=4 sts=4 ts=4 et
