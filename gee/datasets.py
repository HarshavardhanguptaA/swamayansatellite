import ee
from typing import Any
from config.constants import *


def load_sentinel2(aoi: Any, start: str, end: str, cloud: int = 20) -> Any:
    return (
        ee.ImageCollection(SENTINEL_2)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud))
        .select(S2_BANDS)
    )


@st.cache_data
def load_sentinel1_cached(_aoi, start_date, end_date):
    return (
        ee.ImageCollection(SENTINEL_1)
        .filterBounds(_aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(S1_POLARIZATION)
    )



def load_landsat(aoi: Any, start: str, end: str) -> Any:
    return (
        ee.ImageCollection(LANDSAT_8)
        .filterBounds(aoi)
        .filterDate(start, end)
        .select(LANDSAT_OPTICAL + LANDSAT_THERMAL)
    )


def load_rainfall(aoi: Any, start: str, end: str) -> Any:
    return (
        ee.ImageCollection(GPM)
        .filterBounds(aoi)
        .filterDate(start, end)
        .select("precipitationCal")
    )
