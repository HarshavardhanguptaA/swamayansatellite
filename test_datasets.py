import ee

from gee.auth import initialize_earth_engine
from gee.datasets import load_sentinel2   # ✅ THIS WAS MISSING


# initialize Earth Engine
initialize_earth_engine()


# Area of interest
aoi = ee.Geometry.Point([80.2, 13.0]).buffer(10000)

# Date range
start = "2024-01-01"
end = "2024-02-15"


# Load Sentinel-2
collection = load_sentinel2(
    aoi=aoi,
    start=start,
    end=end,
    cloud=60
)

print("Number of images:", collection.size().getInfo())
