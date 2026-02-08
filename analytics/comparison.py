def build_feature_pairs(df, feature_x, feature_y):
    pair = df[['date', feature_x, feature_y]].dropna()
    return pair.sort_values("date")
