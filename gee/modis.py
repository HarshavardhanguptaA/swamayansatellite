import ee


def load_modis_lst(aoi, start_date, end_date):

    modis = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
    )

    def compute_lst(image):

        # -----------------
        # Day LST
        # -----------------
        day = image.select("LST_Day_1km")
        day = day.updateMask(day.neq(0))

        lst_day = (
            day.multiply(0.02)
            .subtract(273.15)
            .rename("LST_Day")
        )

        # -----------------
        # Night LST
        # -----------------
        night = image.select("LST_Night_1km")

        lst_night = (
            night.multiply(0.02)
            .subtract(273.15)
            .rename("LST_Night")
        )

        return (
            image
            .addBands(lst_day)
            .addBands(lst_night)
            .copyProperties(image, ["system:time_start"])
        )

    return modis.map(compute_lst)
