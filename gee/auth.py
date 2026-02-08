import ee


def initialize_earth_engine():
    """
    Initializes Google Earth Engine.
    """
    ee.Initialize(project="multisatellitefusion")
    return ee
