import ee


def load_sentinel1(aoi, start_date, end_date):
    """
    Load Sentinel-1 SAR GRD data.
    """

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .select(["VV", "VH"])
    )

    return collection


def to_db(image):
    """
    Convert linear backscatter to dB.
    """
    return ee.Image(10).multiply(image.log10()).copyProperties(
        image, ["system:time_start"]
    )
