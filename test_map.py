import ee
import geemap

from gee.auth import initialize_earth_engine
from gee.datasets import load_sentinel2

# Initialize Earth Engine
initialize_earth_engine()

# Define AOI
aoi = ee.Geometry.Point([80.2, 13.0]).buffer(15000)

# Load Sentinel-2 images
collection = load_sentinel2(
    aoi=aoi,
    start="2024-01-01",
    end="2024-02-15",
    cloud=60
)

# Reduce to single image
image = collection.median()

# Create interactive map
Map = geemap.Map(center=[13.0, 80.2], zoom=9)

# Visualization parameters
vis = {
    "bands": ["B4", "B3", "B2"],
    "min": 0,
    "max": 3000,
}

# Add to map
Map.addLayer(image, vis, "Sentinel-2 RGB")
Map.addLayer(aoi, {}, "AOI")

# Display map
Map
