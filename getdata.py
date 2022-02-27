import ee
import urllib.request
from PIL import Image
import numpy as np
import requests

ee.Initialize()
era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY')
crop_map = ee.Image('USGS/GFSAD1000_V1')
crop_map = crop_map.select('landcover')
i_date = '2020-07-01'
f_date = '2020-08-01'
era5 = era5.filterDate(i_date, f_date)

bands = ['temperature_2m', "soil_temperature_level_1", "total_precipitation", "surface_net_solar_radiation"]
b_min = [250, 250, 0, 80000]
b_max = [320,320,0.01, 25000000]
TRAINING_COUNT = 10000
DIMENSIONS = 512
LONGITUDE =[(-75, -45), (-125, -75), (-15, 45), (0, 135), (120, 150)]
LATITUDE = [(-45,10), (15, 55), (-30, 40),(15, 60),(-35, -15)]
BUFFER = 100
era5_img = era5.mean()


for count in range(TRAINING_COUNT):
    if count % 10 == 0:
        print(count)
    idx = np.random.randint(5, size = 1)[0]
    longitude, latitude = LONGITUDE[idx], LATITUDE[idx]
    lon, lat = np.random.uniform(*longitude, size=(1,))[0], np.random.uniform(*latitude, size=(1,))[0]
    poi = ee.Geometry.Point(lon, lat)
    roi = poi.buffer(BUFFER)

    gt_info = {
        'min': 0.0,
        'max': 5.0,
        'dimensions': 500,
        'region': roi,
        'palette': ['black', 'black', 'black', 'yellow', 'yellow']
    }

    gt_url = crop_map.getThumbUrl(gt_info)
    r = requests.get(gt_url)
    with open(f'gt_url/{count}.png', 'wb') as f:
        f.write(r.content)

    for i, band in enumerate(bands):
        img_info = {
            'min': b_min[i],
            'max': b_max[i],
            'bands':[band],
            'dimensions': DIMENSIONS,
            'palette': ["000080","#0000D9","#4000FF","#8000FF","#0080FF","#00FFFF",
            "#00FF80","#80FF00","#DAFF00","#FFFF00","#FFF500","#FFDA00",
            "#FFB000","#FFA400","#FF4F00","#FF2500","#FF0A00","#FF00FF"],
            'region': roi, 
        }
        url = era5_img.getThumbUrl(img_info)
        folder_dir = band.split('_')[0]
        r = requests.get(url)
        with open(f'url/{count}.png', 'wb') as f:
            f.write(r.content)

