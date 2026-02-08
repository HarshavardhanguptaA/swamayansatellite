import ee

# -----------------------
# Initialize Earth Engine
# -----------------------
ee.Initialize()

# -----------------------
# Test AOI (your AOI)
# -----------------------
aoi = ee.Geometry.Point([77.2, 28.6]).buffer(3000)
# ↑ Delhi region — guaranteed MODIS data

# -----------------------
# Load MODIS
# -----------------------
modis = (
    ee.ImageCollection("MODIS/061/MOD11A2")
    .filterBounds(aoi)
    .filterDate("2023-01-01", "2024-12-31")
)

print("MODIS image count:", modis.size().getInfo())

# -----------------------
# Check band names
# -----------------------
first = modis.first()
print("Bands:", first.bandNames().getInfo())

# -----------------------
# Extract one pixel value
# -----------------------
sample = first.select("LST_Day_1km").reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=aoi,
    scale=1000,
    maxPixels=1e13
)

print("Raw MODIS LST value:", sample.getInfo())

# -----------------------
# Convert to Celsius
# -----------------------
if sample.getInfo() is not None:
    k = sample.getInfo()["LST_Day_1km"]
    c = k * 0.02 - 273.15
    print("Converted Celsius:", c)
