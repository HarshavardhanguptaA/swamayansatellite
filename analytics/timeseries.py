import ee
import pandas as pd


def extract_timeseries(
    collection,
    band_name,
    region,
    scale=10,
    reducer=None,
):
    """
    Extract time-series safely from an ImageCollection.
    """

    # ✅ define reducer INSIDE function
    if reducer is None:
        reducer = ee.Reducer.mean()

    def image_to_feature(image):
        stats = image.select(band_name).reduceRegion(
            reducer=reducer,
            geometry=region,
            scale=scale,
            maxPixels=1e13,
            bestEffort=True,
        )

        return ee.Feature(
            None,
            {
                "date": image.date().format("YYYY-MM-dd"),
                band_name: stats.get(band_name),
            },
        )

    features = collection.map(image_to_feature)

    info = features.getInfo()

    records = []

    for f in info["features"]:
        props = f["properties"]
        value = props.get(band_name)

        if value is not None:
            records.append(
                {
                    "date": props["date"],
                    band_name: float(value),
                }
            )

    if len(records) == 0:
        return pd.DataFrame(columns=["date", band_name])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df
 