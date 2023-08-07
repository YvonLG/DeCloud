
from typing import *
from pathlib import Path

import pandas as pd
import shapely
import shapely.ops as sops
import pyproj
import numpy as np
import rasterio as rio

from datetime import datetime
import json
from tqdm import tqdm

# TODO: add running mean/std util

def get_path_dicts(path: Path, sanity_check: bool=False) -> List[Dict[str, Path]]:
    """The subfolder architecture must be:
    path/
    ├── time_serie1/
    │   ├── s1.tif
    │   ├── s2.tif
    │   ├── s2cloudless.tif
    │   ├── s1_propeties.csv
    |   ├── s2_properties.csv
    │   └── tile.json
    ├── time_serie2/
    ..."""
    path_dicts = []

    iterdir = list(path.iterdir())
    for dir in tqdm(iterdir) if sanity_check else iterdir:
        if not dir.is_dir():
            continue
        
        path_dict = {
            's1': dir / 's1.tif',
            's2': dir / 's2.tif',
            's2cloudless': dir / 's2cloudless.tif',
            's1_properties': dir / 's1_properties.csv',
            's2_properties': dir / 's2_properties.csv',
            'tile': dir / 'tile.json',
        }

        if sanity_check:
            with rio.open(path_dict['s2'], 'r') as src:
                data = src.read()
            if np.count_nonzero(data == 0) > 200:
                continue

        path_dicts.append(path_dict)

    return path_dicts

def read_dates(properties_path: Path) -> List[int]:
    """Will only work if the property 'system:time_start' is defined."""
    properties = pd.read_csv(properties_path)
    dates =  properties['system:time_start'].tolist()
    return [int(d/1000) for d in dates]


def read_months(properties_path: Path) -> List[int]:
    """Will only work if the property 'system:time_start' is defined.
    Jan-Dec -> 0-11"""
    dates = read_dates(properties_path)
    return [datetime.fromtimestamp(d).month-1 for d in dates]


def read_lat_lon(tile_path: Path) -> Tuple[float, float]:
    with open(tile_path, 'r') as src:
        tile = json.load(src)

    crs = tile['crs']['properties']['name']

    wgs84 = pyproj.CRS('EPSG:4326')
    tile_crs = pyproj.CRS(crs)

    project = pyproj.Transformer.from_crs(tile_crs, wgs84, always_xy=True).transform
    centroid = shapely.geometry.shape(tile).centroid
    centroid = sops.transform(project, centroid)
    lon, lat = centroid.coords.xy
    return lat[0], lon[0]

def clip_and_rescale(arr: np.ndarray, min: float, max: float, nmin: float, nmax: float) -> np.ndarray:
    arr = np.clip(arr, min, max)
    arr = nmin + (nmax - nmin) * (arr - min) / (max - min)
    return arr