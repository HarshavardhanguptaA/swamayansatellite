def add_season(df):
    df = df.copy()
    df["month"] = df["date"].dt.month

    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon"
    })

    return df
