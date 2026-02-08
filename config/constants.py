import ee

# Default study area (can be changed later from UI)
DEFAULT_AOI = aoi = ee.Geometry.Point([78.2, 11.6]).buffer(30000)  # type: ignore


SENTINEL_2 = "COPERNICUS/S2_SR_HARMONIZED"
SENTINEL_1 = "COPERNICUS/S1_GRD"
LANDSAT_8 = "LANDSAT/LC08/C02/T1_L2"
GPM = "NASA/GPM_L3/IMERG"
SRTM = "USGS/SRTMGL1_003"
S2_BANDS = ["B2", "B3", "B4", "B8"]
S1_POLARIZATION = ["VV", "VH"]
LANDSAT_OPTICAL = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
LANDSAT_THERMAL = ["ST_B10"]
