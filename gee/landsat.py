import ee


# -----------------------------
# Cloud mask for Landsat Level-1
# -----------------------------
def mask_landsat_l1(image):
    qa = image.select("QA_PIXEL")

    cloud = qa.bitwiseAnd(1 << 3).eq(0)
    shadow = qa.bitwiseAnd(1 << 4).eq(0)

    return image.updateMask(cloud.And(shadow))


# ------------------------------------------------
# Brightness temperature → Land Surface Temperature
# ------------------------------------------------
def compute_lst_from_bt(image):

    k1 = ee.Image.constant(image.get("K1_CONSTANT_BAND_10"))
    k2 = ee.Image.constant(image.get("K2_CONSTANT_BAND_10"))

    radiance = image.select("B10").multiply(0.1)

    bt = k2.divide(
        (k1.divide(radiance)).add(1).log()
    )

    lst = bt.subtract(273.15).rename("LST")

    return image.addBands(lst).copyProperties(
        image, ["system:time_start"]
    )



    lst_celsius = bt.subtract(273.15).rename("LST")

    return image.addBands(lst_celsius).copyProperties(
        image, ["system:time_start"]
    )


# -----------------------------
# Landsat loader
# -----------------------------
def load_landsat(aoi, start_date, end_date):

    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    landsat = l8.merge(l9)

    landsat = (
    landsat
    .map(compute_lst_from_bt)
    .map(mask_landsat_l1)
)


    return landsat
