#!/usr/bin/env python3
# Generate file with street points at regular intervals.
#
# Example:
#
# python3 make_street_points.py -c myconfig.json -S 28992 -I 50 -o amsterdam_points.geojson
# 
# where 'myconfig.json' contains:
# {
#  "bounding_box": {
#    "west": 4.7149,
#    "south": 52.2818,
#    "east": 5.1220,
#    "north": 52.4284
# },
# ...
# }
#
# See the examples/ directory for configfile format; it is the same as used by
# the script mapillary_jpg_download.py
#
################################################################################
# This script is provided as-is. The usage of this script, compliance with
# Mapillary licencing and acceptable use terms, as well as any Internet service
# provider terms, is entirely your responsibility.
#
# Licence:
#   Copyright 2024 Matthew Danish
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0

import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, LineString
import numpy as np
from pathlib import Path
import argparse
import json
import sys

parser = argparse.ArgumentParser(prog='make_street_points.py', description='Generate geo file with points at given interval along streets in bbox region')
parser.add_argument('--configfile', '--config', '-c', required=True, metavar='FILENAME', help='Configuration file to process (see examples/ dir)')
parser.add_argument('--output', '-o', metavar='FILENAME', required=True, help='Write points into FILENAME (format depending on extension, e.g. .geojson)')
parser.add_argument('--srid', '-S', required=True, metavar='NUM', type=int, help='SRID of the coordinate reference system (e.g. 28992 for The Netherlands)')
parser.add_argument('--interval', '-I', required=True, metavar='NUM', type=int, help='Approximate number of meters in between each point (on a given line segment)')

def generate_street_points(bbox, spacing=50, srid=28992):
    """
    bbox: tuple of (north, south, east, west)
    spacing: distance between points in meters
    """
    # Download street network
    north, south, east, west = bbox
    G = ox.graph_from_bbox(bbox, network_type='all')
    
    # Convert to GeoDataFrame
    edges = ox.graph_to_gdfs(G, nodes=False).to_crs(srid)
    
    points = []
    for _, row in edges.iterrows():
        line = row['geometry']
        # Calculate number of points needed
        line_length = line.length
        num_points = int(line_length / spacing)
        
        if num_points > 0:
            # Generate points at regular intervals
            distances = np.linspace(0, line_length, num_points + 1)
            for distance in distances:
                point = line.interpolate(distance)
                points.append(Point(point.x, point.y))
    
    # Create GeoDataFrame of points
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=edges.crs)
    
    return points_gdf.to_crs(4326)

def fast_deduplicate_points(points_gdf, distance_threshold=50, srid=28992):
   """Deduplicates points using a grid-based approach"""
   if not points_gdf.crs or points_gdf.crs.is_geographic:
       points = points_gdf.to_crs(srid)
   else:
       points = points_gdf.copy()
   
   # Create grid cells for points
   minx, miny, maxx, maxy = points.total_bounds
   cell_size = distance_threshold
   
   # Assign cell indices to points
   points['cell_x'] = ((points.geometry.x - minx) / cell_size).astype(int)
   points['cell_y'] = ((points.geometry.y - miny) / cell_size).astype(int)
   points['cell'] = points.apply(lambda row: f"{row.cell_x}_{row.cell_y}", axis=1)
   
   # Keep first point in each cell
   deduped = points.drop_duplicates(subset='cell')
   
   # Cleanup and convert back to GeoDataFrame
   deduped = deduped.drop(['cell_x', 'cell_y', 'cell'], axis=1)
   
   # Return to original CRS if needed
   if points_gdf.crs != deduped.crs:
       deduped = deduped.to_crs(points_gdf.crs)
   
   return deduped

def main():
    args = parser.parse_args()

    with open(args.configfile) as f:
        config = json.load(f)

    b = config['bounding_box']
    if int(ox.__version__.split('.')[0]) < 2:
        print(f'Detected OSMnx version {ox.__version__} < 2: not supported')
        sys.exit(1)
    else:
        print(f'Detected OSMnx version {ox.__version__} >= 2. Ok.')
        bbox = (b['west'], b['south'], b['east'], b['north'])
    srid = args.srid
    output_file = args.output

    points = generate_street_points(bbox, args.interval, srid)

    points2 = fast_deduplicate_points(points, args.interval, srid)

    points2.to_file(output_file)

if __name__=='__main__':
    main()
