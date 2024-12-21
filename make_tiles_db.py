#!/usr/bin/env python3
import argparse
import json
import sys
import os
from pathlib import Path
import pickle
import lzma
import imagesize

parser = argparse.ArgumentParser(prog='make_tiles_db.py', description='Create single compressed database file with information from tiles JSON files.')
parser.add_argument('dir', metavar='DIR', help='Directory of tiles JSON files')
parser.add_argument('--output', '-o', metavar='FILENAME', required=True, help='Write tile database into given FILENAME')
parser.add_argument('--seqs', type=str, help='Sequences dir (where we can find <seqid>/<imgid>.jpg image files for analysis)', default=None)

def main():
    args = parser.parse_args()
    tiles = Path(args.dir)
    db = {}
    seqs = Path(args.seqs) if args.seqs else None
    seqs = seqs if seqs and seqs.exists() else None

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
            if seqs:
                imagefile = seqs / Path(seqid) / Path(str(imgid)).with_suffix('.jpg')
                if imagefile.exists():
                    w, h = imagesize.get(imagefile)
                    db[imgid]['image_width'] = w
                    db[imgid]['image_height'] = h
            #vlog(json.dumps(db[imgid]))

    for tilefile in tiles.glob('mly1_public*'):
        with tilefile.open() as fp:
            tilesJson = json.load(fp)
            processTilesJson(tilesJson)
    
    with lzma.open(args.output, 'wb') as fp:
        print(f'Writing tiles database with {len(db)} entries to: {args.output}')
        pickle.dump(db, fp)


if __name__=='__main__':
    main()
# vim: ai sw=4 sts=4 ts=4 et
