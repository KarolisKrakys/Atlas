from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
import json

# SERVICE_ACCOUNT = '791669903945-73sk1gn6vkvn742mumd6ip8hqovqcmoj.apps.googleusercontent.com'
SERVICE_ACCOUNT = 'crop-data@hackathon-342514.iam.gserviceaccount.com'
KEY = 'key.json'
PROJECT = '2022-hackathon'
credentials = service_account.Credentials.from_service_account_file(KEY)
scoped_credentials = credentials.with_scopes(
    ['https://www.googleapis.com/auth/cloud-platform'])

session = AuthorizedSession(scoped_credentials)

url = 'https://earthengine.googleapis.com/v1beta/projects/earthengine-public/assets/LANDSAT'

response = session.get(url)


import urllib
import cv2 as cv
from PIL import Image
import numpy as np 
import io
import ee
# project = 'projects/earthengine-public'
# asset_id = 'COPERNICUS/S2/20170430T190351_20170430T190351_T10SEG'
# name = '{}/assets/{}'.format(project, asset_id)
# url = 'https://earthengine.googleapis.com/v1alpha/{}'.format(name)

# response = session.get(url)
# content = response.content

# asset = json.loads(content)
# url = 'https://earthengine.googleapis.com/v1alpha/{}:getPixels'.format(name)
# body = json.dumps({
#     'fileFormat': 'PNG',
#     'bandIds': ['B4', 'B3', 'B2'],
#     'region': asset['geometry'],
#     'grid': {
#         'dimensions': {'width': 256, 'height': 256},
#     },
#     'visualizationOptions': {
#         'ranges': [{'min': 0, 'max': 3000}],
#     },
# })

# image_response = session.post(url, body)
# image_content = image_response.content

# image = Image.open(io.BytesIO(image_content))
# image.show()


ee_creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY)
ee.Initialize(ee_creds)

coords = [
  -121.58626826832939,
  38.059141484827485,
]
region = ee.Geometry.Point(coords)

collection = ee.ImageCollection('COPERNICUS/S2')
collection = collection.filterBounds(region)
collection = collection.filterDate('2020-04-01', '2020-09-01')
image = collection.median()

serialized = ee.serializer.encode(image)

# Make a projection to discover the scale in degrees.
proj = ee.Projection('EPSG:4326').atScale(10).getInfo()

# Get scales out of the transform.
scale_x = proj['transform'][0]
scale_y = -proj['transform'][4]

url = 'https://earthengine.googleapis.com/v1beta/projects/{}/image:computePixels'
url = url.format(PROJECT)

response = session.post(
  url=url,
  data=json.dumps({
    'expression': serialized,
    'fileFormat': 'PNG',
    'bandIds': ['B4','B3','B2'],
    'grid': {
      'dimensions': {
        'width': 640,
        'height': 640
      },
      'affineTransform': {
        'scaleX': scale_x,
        'shearX': 0,
        'translateX': coords[0],
        'shearY': 0,
        'scaleY': scale_y,
        'translateY': coords[1]
      },
      'crsCode': 'EPSG:4326',
    },
    'visualizationOptions': {'ranges': [{'min': 0, 'max': 3000}]},
  })
)

image_content = response.content

image = Image.open(io.BytesIO(image_content))
image.show()