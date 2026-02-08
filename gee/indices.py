import ee
from typing import Any

# ----------------------------------------------------
# Sentinel-2 spectral bands
# ----------------------------------------------------
S2_BANDS = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    "NIR": "B8",
}


# ----------------------------------------------------
# NDVI
# ----------------------------------------------------
def add_ndvi(image: Any) -> Any:
    """
    Adds NDVI band to an image.
    NDVI = (NIR - RED) / (NIR + RED)
    """

    ndvi = image.normalizedDifference(
        [S2_BANDS["NIR"], S2_BANDS["RED"]]
    ).rename("NDVI")

    return image.addBands(ndvi)


def compute_ndvi(collection: Any) -> Any:
    """
    Computes NDVI for an ImageCollection.
    """
    return collection.map(add_ndvi)


# ----------------------------------------------------
# NDWI
# ----------------------------------------------------
def add_ndwi(image: Any) -> Any:
    """
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    """

    ndwi = image.normalizedDifference(
        [S2_BANDS["GREEN"], S2_BANDS["NIR"]]
    ).rename("NDWI")

    return image.addBands(ndwi)


def compute_ndwi(collection: Any) -> Any:
    return collection.map(add_ndwi)


# ----------------------------------------------------
# EVI
# ----------------------------------------------------
def add_evi(image: Any) -> Any:
    """
    EVI = 2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)
    """

    nir = image.select(S2_BANDS["NIR"])
    red = image.select(S2_BANDS["RED"])
    blue = image.select(S2_BANDS["BLUE"])

    evi = (
        nir.subtract(red)
        .multiply(2.5)
        .divide(
            nir.add(red.multiply(6))
            .subtract(blue.multiply(7.5))
            .add(1)
        )
        .rename("EVI")
    )

    return image.addBands(evi)


def compute_evi(collection: Any) -> Any:
    return collection.map(add_evi)