import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ee
import geemap
import streamlit as st
import plotly.express as px
import pandas as pd
# =========================
# CACHED EARTH ENGINE LOADERS
# =========================

@st.cache_data(show_spinner=True)
def load_sentinel2_cached(_aoi, start_date, end_date):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(_aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B2", "B3", "B4", "B8"])
    )
    s2 = compute_ndvi(s2)
    s2 = compute_ndwi(s2)
    s2 = compute_evi(s2)
    return s2


@st.cache_data(show_spinner=True)
def load_sentinel1_cached(_aoi, start_date, end_date):
    return load_sentinel1(_aoi, start_date, end_date).map(to_db)


@st.cache_data(show_spinner=True)
def load_modis_cached(_aoi, start, end):
    return load_modis_lst(_aoi, start, end)


def cached_timeseries(collection, band, region, scale):
    return extract_timeseries(collection, band, region, scale)




if "ee_initialized" not in st.session_state:
    ee.Initialize(project="multisatellitefusion")
    st.session_state.ee_initialized = True


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("SWAYAMYAN")

# =========================
# AOI INPUT
# =========================

st.sidebar.header("Analysis Mode")

satellite = st.sidebar.radio(
    "Select Satellite",
    ["Sentinel-2 (Optical)", "Sentinel-1 (SAR)", "MODIS (Thermal)"]
)

st.sidebar.header("Area of Interest (AOI)")

aoi_input = st.sidebar.text_input(
    "Enter AOI (min_lon, min_lat, max_lon, max_lat)",
    value="77.5, 12.8, 77.7, 13.0",
)

def parse_aoi(aoi_str):
    try:
        parts = [float(x.strip()) for x in aoi_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return parts
    except Exception:
        st.error("Invalid AOI format. Use: min_lon, min_lat, max_lon, max_lat")
        return None

aoi_coords = parse_aoi(aoi_input)

if aoi_coords is None:
    st.stop()

aoi = ee.Geometry.Rectangle(aoi_coords)

from analytics.statistics import compute_dtr, correlation_matrix
from ui.charts import correlation_heatmap


from gee.sentinel1 import load_sentinel1, to_db
from gee.modis import load_modis_lst
from gee.indices import compute_ndvi, compute_ndwi, compute_evi
from analytics.timeseries import extract_timeseries


lag_dict = None
fig_lag = None
lag_df = None
feature_x = None
feature_y = None
stability = None

def normalize_feature_df(df, feature_name):
    """
    Ensures dataframe has exactly:
    date | feature_name
    """
    if df is None or df.empty:
        return None

    df = df.copy()
    df = df.loc[:, ~df.columns.duplicated()]

    cols = list(df.columns)

    if "date" not in cols:
        return None

    value_cols = [c for c in cols if c != "date"]

    if len(value_cols) != 1:
        return None  

    df = df[["date", value_cols[0]]]
    df.columns = ["date", feature_name]

    df["date"] = pd.to_datetime(df["date"])

    df = (
        df.groupby("date", as_index=False)
          .mean()
    )

    return df.set_index("date")


def clean_feature_df(df):
    """
    Ensures dataframe has exactly:
    date | feature
    """
    if df is None or df.empty:
        return df

    df = df.loc[:, ~df.columns.duplicated()]

    if "date" in df.columns:
        cols = ["date"] + [c for c in df.columns if c != "date"]
        df = df[cols]

    return df

def monthly_aggregate(df):
    """
    Converts date-level dataframe to monthly means.
    Expects:
        date | feature1 | feature2 | ...
    Returns:
        month | feature1 | feature2 | ...
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly_df = (
        df.drop(columns="date")
          .groupby("month")
          .mean()
          .reset_index()
          .rename(columns={"month": "date"})
    )

    return monthly_df

ee.Initialize(project="multisatellitefusion")

aoi = ee.Geometry.Point([78.2, 11.6]).buffer(30000)

Map = geemap.Map(center=[11.6, 78.2], zoom=7)

s2 = load_sentinel2_cached(
    aoi,                 # positional argument ONLY
    "2024-01-01",
    "2024-01-31"
)




rgb = s2.median().clip(aoi)

Map.addLayer(
    rgb,
    {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
    "Sentinel-2 RGB",
)

Map.to_streamlit(height=500)
# =========================
# GLOBAL FEATURE HOLDERS
# =========================
ndvi_df = None
ndwi_df = None
evi_df = None
vv_df = None
vh_df = None
lst_day_df = None
lst_night_df = None

# =========================
# OPTION 1 — SINGLE-SATELLITE ANALYTICS
# =========================

if satellite == "Sentinel-2 (Optical)":

    st.header("Sentinel-2 Optical Analytics")
    
    s2 = load_sentinel2_cached(
        aoi,                 # positional argument ONLY
        "2024-01-01",
        "2024-01-31"
)


    ndvi_df = cached_timeseries(s2, "NDVI", aoi, 10)
    ndwi_df = cached_timeseries(s2, "NDWI", aoi, 10)
    evi_df  = cached_timeseries(s2, "EVI",  aoi, 10)

    feature = st.selectbox(
        "Select Feature",
        ["NDVI", "NDWI", "EVI"],
        key="s2_feature_select_optical"
)



    df = {"NDVI": ndvi_df, "NDWI": ndwi_df, "EVI": evi_df}[feature]

    st.subheader(f"{feature} Time Series Data")
    st.dataframe(df)

    show_plot = st.checkbox("Show time-series plot")

    if show_plot and df is not None and not df.empty:
        st.plotly_chart(
            px.line(df, x="date", y=feature, title=f"{feature} Time Series"),
            use_container_width=True
        )


elif satellite == "Sentinel-1 (SAR)":

    st.header("Sentinel-1 SAR Analytics")
    s1_collection = load_sentinel1_cached(
        aoi,
        "2024-01-01",
        "2024-01-31"
)

    vv_df = extract_timeseries(s1_collection, "VV", aoi, 10)
    vh_df = extract_timeseries(s1_collection, "VH", aoi, 10)

    feature = st.selectbox(
        "Select Feature",
        ["VV", "VH"],
        key="s1_feature_select_sar"
)

    df = vv_df if feature == "VV" else vh_df

    st.subheader(f"{feature} Backscatter (dB)")
    st.dataframe(df)

    show_plot = st.checkbox("Show time-series plot")

    if show_plot and df is not None and not df.empty:
        st.plotly_chart(
            px.line(df, x="date", y=feature, title=f"{feature} Time Series"),
            use_container_width=True
        )


elif satellite == "MODIS (Thermal)":

    st.header("MODIS Thermal Analytics")

    modis = load_modis_cached(
    aoi,
    "2023-01-01",
    "2024-12-31"
)


    lst_day_df = extract_timeseries(
        modis, "LST_Day", aoi.buffer(3000), 1000
    )

    lst_night_df = extract_timeseries(
        modis, "LST_Night", aoi.buffer(3000), 1000
    )

    feature = st.selectbox(
        "Select Feature",
        ["LST_Day", "LST_Night"],
        key="modis_feature_select_thermal"
)

    df = lst_day_df if feature == "LST_Day" else lst_night_df

    st.subheader(f"{feature} (°C)")
    st.dataframe(df)

    show_plot = st.checkbox("Show time-series plot")

    if show_plot and df is not None and not df.empty:
        st.plotly_chart(
            px.line(df, x="date", y=feature, title=f"{feature} Time Series"),
            use_container_width=True
        )


from analytics.lag_analysis import lag_correlation
from ui.charts import lag_correlation_plot
from analytics.fusion_readiness import (
    detect_peak_lag,
    stability_score,
    fusion_decision
)

import pandas as pd
import plotly.express as px
import streamlit as st

lag_dict = None
lag_df = None
feature_x = None
feature_y = None
fig_lag = None
fusion_x = None
fusion_y = None


# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Visualization", "Analysis", "Fusion Readiness"])

# =========================
# TAB 2 — ANALYSIS
# =========================
# =========================
# TAB 2 — CROSS-SATELLITE ANALYSIS
# =========================
# =========================
# TAB 2 — CROSS-SATELLITE ANALYSIS
# =========================
with tab2:

    st.subheader("Cross-Satellite Analysis")

    # -------------------------
    # Satellite selection
    # -------------------------
    analysis_satellites = st.multiselect(
        "Select satellites for analysis",
        ["Sentinel-2", "Sentinel-1", "MODIS"],
        default=["Sentinel-2", "Sentinel-1"]
    )

    if len(analysis_satellites) < 2:
        st.info("Select at least two satellites for analysis.")
        st.stop()

    # -------------------------
    # Load features dynamically
    # -------------------------
    feature_dfs = []

    # ---------- Sentinel-2 ----------
    if "Sentinel-2" in analysis_satellites:
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
           .filterDate("2023-01-01", "2024-12-31")

            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(["B2", "B3", "B4", "B8"])
        )
        s2 = compute_ndvi(s2)
        ndvi = extract_timeseries(s2, "NDVI", aoi, 10)
        if ndvi is not None and not ndvi.empty:
            feature_dfs.append(ndvi)

    # ---------- Sentinel-1 ----------
    if "Sentinel-1" in analysis_satellites:
        s1 = load_sentinel1_cached(
            aoi,
            "2023-01-01",
            "2024-12-31"
    )

    vv = cached_timeseries(s1, "VV", aoi, 10)

    if vv is not None and not vv.empty:
        feature_dfs.append(vv)

    # ---------- MODIS ----------
    if "MODIS" in analysis_satellites:
        modis = load_modis_lst(aoi, "2023-01-01", "2024-12-31")

        lst_day = extract_timeseries(
            modis, "LST_Day", aoi.buffer(3000), 1000
        )
        lst_night = extract_timeseries(
            modis, "LST_Night", aoi.buffer(3000), 1000
        )

        for df_modis in [lst_day, lst_night]:
            if df_modis is not None and not df_modis.empty:
                feature_dfs.append(df_modis)

    # -------------------------
    # Normalize features
    # -------------------------
    normalized = []

    for raw_df in feature_dfs:
        feature_name = raw_df.columns[1]
        norm = normalize_feature_df(raw_df, feature_name)
        if norm is not None:
            normalized.append(norm)

    if len(normalized) < 2:
        st.warning("Unable to normalize enough features.")
        st.stop()

    # -------------------------
    # Temporal alignment
    # -------------------------
    temporal_mode = st.radio(
        "Temporal alignment",
        ["Original (sensor dates)", "Monthly"],
        horizontal=True
    )

    # 🔒 FORCE UNIQUE INDICES BEFORE CONCAT
    safe_normalized = []

    for d in normalized:
        d = d.copy()

        # ensure datetime index
        d.index = pd.to_datetime(d.index)

    # collapse duplicate dates
        d = d.groupby(d.index).mean()

        safe_normalized.append(d)

    df = pd.concat(safe_normalized, axis=1).reset_index()


    if temporal_mode == "Monthly":
        df = monthly_aggregate(df)

    # ✅ Correct overlap check (AFTER alignment)
    if df.empty or df.shape[1] < 3:
        st.warning(
            "No overlapping observations after temporal alignment. "
            "Try Monthly aggregation or select different satellites."
        )
        st.stop()

    # -------------------------
    # Correlation
    # -------------------------
    method = st.radio(
        "Correlation method",
        ["pearson", "spearman"],
        horizontal=True
    )

    corr_df = correlation_matrix(df, method=method)
    fig_corr = correlation_heatmap(
        corr_df,
        title=f"{method.title()} Correlation Between Satellite Features"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # -------------------------
    # Lag analysis
    # -------------------------
    st.subheader("Lag Relationship Analysis")

    col1, col2 = st.columns(2)

    with col1:
        feature_x = st.selectbox(
            "Driver feature (X)",
            df.columns.drop("date"),
            key="lag_feature_x"
)

    with col2:
       feature_y = st.selectbox(
           "Response feature (Y)",
           df.columns.drop("date"),
           key="lag_feature_y"
)

    if feature_x == feature_y:
        st.warning("Select two different features.")
        st.stop()

    lag_df = df[["date", feature_x, feature_y]].dropna()

    if len(lag_df) < 2:
        st.warning("Not enough overlapping observations.")
        st.stop()

    max_lag = st.slider(
        "Maximum lag (timesteps)",
        min_value=1,
        max_value=min(12, len(lag_df) // 2),
        value=3
    )

    lag_dict = lag_correlation(
        lag_df[feature_x].values,
        lag_df[feature_y].values,
        max_lag=max_lag
    )

    fig_lag = lag_correlation_plot(
        lag_dict,
        title=f"{feature_x} → {feature_y} Lag Correlation"
    )

    st.plotly_chart(fig_lag, use_container_width=True)

    # 🔒 Save aligned dataframe for Fusion
    st.session_state["analysis_df"] = df.copy()


with tab3:
    
    st.subheader("Feature Pair Selection")
    available_features = [
    col for col in df.columns if col != "date"
]

if len(available_features) < 2:
    st.warning("Not enough features available for fusion.")
    st.stop()


    st.subheader("Fusion Readiness & Feature-Level Fusion")

    if lag_dict is None or lag_df is None:
        st.info("Run lag analysis in the Analysis tab before performing fusion.")
        st.stop()

    
    peak_lag, peak_corr = detect_peak_lag(lag_dict)

    stability = stability_score(
        lag_df[feature_x].values,
        lag_df[feature_y].values
    )

    if peak_lag is None or peak_corr is None:
        st.warning("Fusion readiness could not be assessed.")
        st.stop()
        
# ---------- PEAK LAG EXTRACTION ----------
if lag_dict is None:
    st.warning("Run lag analysis in the Analysis tab before fusion.")
    st.stop()

peak_lag, peak_corr = detect_peak_lag(lag_dict)

if peak_lag is None or peak_corr is None:
    st.warning("Unable to determine peak lag for fusion.")
    st.stop()


    # ---------- FUSION METRICS ----------
colA, colB, colC = st.columns(3)
colA.metric("Peak Lag (τ)", peak_lag)
colB.metric("Correlation Strength", round(peak_corr, 3))
if stability is not None:
    colC.metric("Stability Score", round(stability, 3))


st.success(
    fusion_decision(
        corr=peak_corr,
        lag=peak_lag,
        stability=stability
    )
)

# ---------- FUSION CONTROL ----------
st.divider()
st.subheader("Fusion Control")

enable_fusion = st.toggle(
    "Enable feature-level fusion",
    value=False
)

if not enable_fusion:
    st.info("Enable fusion to generate fused feature.")
    st.stop()

if temporal_mode != "Monthly":
    st.info(
        "Cross-satellite fusion requires temporal harmonization. "
        "Switch to Monthly aggregation to enable multi-satellite fusion."
    )


# ---------- FEATURE PAIR SELECTION ----------
st.subheader("Select Features for Fusion")
if fusion_x is not None and fusion_y is not None:
    st.caption(
        f"Fusion pair: {fusion_x} (Satellite A) ↔ {fusion_y} (Satellite B)"
    )



available_features = [c for c in df.columns if c != "date"]

fuse_col1, fuse_col2 = st.columns(2)

with fuse_col1:
    fusion_x = st.selectbox(
        "Fusion Feature A",
        available_features,
        key="fusion_x"
    )

with fuse_col2:
    fusion_y = st.selectbox(
        "Fusion Feature B",
        [f for f in available_features if f != fusion_x],
        key="fusion_y"
    )

# ---------- OVERLAP VALIDATION ----------
fusion_df = df[["date", fusion_x, fusion_y]].dropna()

if fusion_df.shape[0] < 4:
    st.warning(
        "Not enough overlapping observations for fusion. "
        "Try Monthly aggregation or select a different feature pair."
    )
    st.stop()

# ---------- FEATURE-LEVEL FUSION ----------
x_norm = (fusion_df[fusion_x] - fusion_df[fusion_x].mean()) / fusion_df[fusion_x].std()
y_norm = (fusion_df[fusion_y] - fusion_df[fusion_y].mean()) / fusion_df[fusion_y].std()

w1 = abs(peak_corr)
w2 = 1 - w1

fusion_df["FUSED_FEATURE"] = w1 * x_norm + w2 * y_norm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import ee
import geemap.foliumap as geemap
import streamlit as st
import plotly.express as px
import pandas as pd

# =========================
# EARTH ENGINE INIT (ONCE)
# =========================
if "ee_initialized" not in st.session_state:
    try:
        ee.Initialize(project="multisatellitefusion")
        st.session_state.ee_initialized = True
    except Exception:
        ee.Authenticate()
        ee.Initialize(project="multisatellitefusion")
        st.session_state.ee_initialized = True

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("SWAYAMYAN")

# =========================
# AOI INPUT
# =========================
st.sidebar.header("Analysis Mode")

satellite = st.sidebar.radio(
    "Select Satellite",
    ["Sentinel-2 (Optical)", "Sentinel-1 (SAR)", "MODIS (Thermal)"],
    key="satellite_selector"
)


st.sidebar.header("Area of Interest (AOI)")

aoi_input = st.sidebar.text_input(
    "Enter AOI (min_lon, min_lat, max_lon, max_lat)",
    value="77.5, 12.8, 77.7, 13.0",
    key="aoi_input_box"
)

def parse_aoi(aoi_str):
    try:
        parts = [float(x.strip()) for x in aoi_str.split(",")]
        if len(parts) != 4:
            raise ValueError
        return parts
    except Exception:
        st.error("Invalid AOI format. Use: min_lon, min_lat, max_lon, max_lat")
        return None

aoi_coords = parse_aoi(aoi_input)
if aoi_coords is None:
    st.stop()

aoi = ee.Geometry.Rectangle(aoi_coords)

# =========================
# IMPORTS
# =========================
from gee.sentinel1 import load_sentinel1, to_db
from gee.modis import load_modis_lst
from gee.indices import compute_ndvi, compute_ndwi, compute_evi
from analytics.timeseries import extract_timeseries
from analytics.statistics import compute_dtr, correlation_matrix
from ui.charts import correlation_heatmap, lag_correlation_plot
from analytics.lag_analysis import lag_correlation
from analytics.fusion_readiness import detect_peak_lag, stability_score, fusion_decision

# =========================
# CACHED LOADERS (FIXED)
# =========================
@st.cache_data(show_spinner=True)
def load_sentinel2_cached(_aoi, start_date, end_date):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(_aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B2", "B3", "B4", "B8"])
    )
    s2 = compute_ndvi(s2)
    s2 = compute_ndwi(s2)
    s2 = compute_evi(s2)
    return s2

@st.cache_data(show_spinner=True)
def load_sentinel1_cached(_aoi, start_date, end_date):
    return load_sentinel1(_aoi, start_date, end_date).map(to_db)

@st.cache_data(show_spinner=True)
def load_modis_cached(_aoi, start, end):
    return load_modis_lst(_aoi, start, end)


# =========================
# MAP
# =========================
Map = geemap.Map(center=[(aoi_coords[1] + aoi_coords[3]) / 2,
                          (aoi_coords[0] + aoi_coords[2]) / 2], zoom=8)

s2_rgb = load_sentinel2_cached(aoi, "2024-01-01", "2024-01-31")
rgb = s2_rgb.median().clip(aoi)

Map.addLayer(
    rgb,
    {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000},
    "Sentinel-2 RGB",
)
Map.to_streamlit(height=450)

# =========================
# GLOBAL FEATURE HOLDERS
# =========================
ndvi_df = ndwi_df = evi_df = None
vv_df = vh_df = None
lst_day_df = lst_night_df = None

# =========================
# OPTION 1 — SINGLE SATELLITE
# =========================
if satellite == "Sentinel-2 (Optical)":
    s2 = load_sentinel2_cached(aoi, "2024-01-01", "2024-01-31")
    st.write("cached_timeseries comes from:", cached_timeseries.__module__)
    ndvi_df = cached_timeseries(s2, "NDVI", aoi, 10)
    ndwi_df = cached_timeseries(s2, "NDWI", aoi, 10)
    evi_df  = cached_timeseries(s2, "EVI",  aoi, 10)

    feature = st.selectbox(
    "Select Feature",
    ["NDVI", "NDWI", "EVI"],
    key="s2_feature_select"
)


elif satellite == "Sentinel-1 (SAR)":
    s1 = load_sentinel1_cached(aoi, "2024-01-01", "2024-01-31")
    vv_df = cached_timeseries(s1, "VV", aoi, 10)
    vh_df = cached_timeseries(s1, "VH", aoi, 10)

    feature = st.selectbox("Select Feature", ["VV", "VH"])
    df = vv_df if feature == "VV" else vh_df
    st.dataframe(df)

elif satellite == "MODIS (Thermal)":
    modis = load_modis_cached(aoi, "2023-01-01", "2024-12-31")
    lst_day_df = cached_timeseries(modis, "LST_Day", aoi.buffer(3000), 1000)
    lst_night_df = cached_timeseries(modis, "LST_Night", aoi.buffer(3000), 1000)

    feature = st.selectbox("Select Feature", ["LST_Day", "LST_Night"])
    df = lst_day_df if feature == "LST_Day" else lst_night_df
    st.dataframe(df)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Visualization", "Analysis", "Fusion Readiness"])

# =========================
# TAB 2 — CROSS SATELLITE
# =========================
with tab2:
    st.subheader("Cross-Satellite Analysis")

    analysis_satellites = st.multiselect(
        "Select satellites",
        ["Sentinel-2", "Sentinel-1", "MODIS"],
        default=["Sentinel-2", "Sentinel-1"]
    )

    feature_dfs = []

    if "Sentinel-2" in analysis_satellites:
        s2 = load_sentinel2_cached(aoi, "2023-01-01", "2024-12-31")
        ndvi_df = cached_timeseries(s2, "NDVI", aoi, 10)

        if ndvi is not None and not ndvi.empty:
            feature_dfs.append(ndvi)

    if "Sentinel-1" in analysis_satellites:
        s1 = load_sentinel1_cached(aoi, "2023-01-01", "2024-12-31")
        vv_df = cached_timeseries(s1, "VV", aoi, 10)
        if vv_df is not None and not vv_df.empty:
            feature_dfs.append(vv_df)

    if "MODIS" in analysis_satellites:
        modis = load_modis_cached(aoi, "2023-01-01", "2024-12-31")
        for band in ["LST_Day", "LST_Night"]:
            dfm = cached_timeseries(modis, band, aoi.buffer(3000), 1000)
            if dfm is not None and not dfm.empty:
                feature_dfs.append(dfm)

    normalized = []
    for raw in feature_dfs:
        name = raw.columns[1]
        ndf = raw[["date", name]].dropna()
        ndf["date"] = pd.to_datetime(ndf["date"])
        normalized.append(ndf.set_index("date"))

    if len(normalized) < 2:
        st.warning("Not enough data for analysis.")
        st.stop()

    temporal_mode = st.radio("Temporal alignment", ["Original", "Monthly"])
    
    safe_normalized = []
    for d in normalized:
        d = d.copy()

    # ensure datetime index
        d.index = pd.to_datetime(d.index)

    # collapse duplicate dates
        d = d.groupby(d.index).mean()

        safe_normalized.append(d)

    df = pd.concat(safe_normalized, axis=1).reset_index()


    if temporal_mode == "Monthly":
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        df = df.groupby("date").mean().reset_index()

    corr_df = correlation_matrix(df, method="pearson")
    st.plotly_chart(correlation_heatmap(corr_df), use_container_width=True)

    st.subheader("Lag Relationship Analysis")

    feature_x = st.selectbox("Driver (X)", df.columns.drop("date"))
    feature_y = st.selectbox("Response (Y)", [c for c in df.columns if c not in ["date", feature_x]])

    lag_df = df[["date", feature_x, feature_y]].dropna()
    if lag_df.shape[0] < 2:
        st.warning("Not enough overlap.")
        st.stop()

    lag_dict = lag_correlation(lag_df[feature_x].values, lag_df[feature_y].values, max_lag=3)
    st.plotly_chart(lag_correlation_plot(lag_dict), use_container_width=True)

    st.session_state["analysis_df"] = df.copy()

# =========================
# TAB 3 — FUSION
# =========================
with tab3:
    if "analysis_df" not in st.session_state:
        st.info("Run analysis first.")
        st.stop()

    df = st.session_state["analysis_df"]
    available = [c for c in df.columns if c != "date"]

    fusion_x = st.selectbox("Feature A", available)
    fusion_y = st.selectbox("Feature B", [c for c in available if c != fusion_x])

    fusion_df = df[["date", fusion_x, fusion_y]].dropna()
    if fusion_df.shape[0] < 4:
        st.warning("Not enough overlap for fusion.")
        st.stop()

    lag_dict = lag_correlation(
        fusion_df[fusion_x].values,
        fusion_df[fusion_y].values,
        max_lag=3
    )

    peak_lag, peak_corr = detect_peak_lag(lag_dict)
    stability = stability_score(
        fusion_df[fusion_x].values,
        fusion_df[fusion_y].values
    )

    colA, colB, colC = st.columns(3)
    colA.metric("Peak Lag", peak_lag)
    colB.metric("Correlation", round(peak_corr, 3))
    if stability is not None:
        colC.metric("Stability", round(stability, 3))
    else:
        colC.metric("Stability", "—")

    x_norm = (fusion_df[fusion_x] - fusion_df[fusion_x].mean()) / fusion_df[fusion_x].std()
    y_norm = (fusion_df[fusion_y] - fusion_df[fusion_y].mean()) / fusion_df[fusion_y].std()

    fusion_df["FUSED_FEATURE"] = abs(peak_corr) * x_norm + (1 - abs(peak_corr)) * y_norm

    st.plotly_chart(
        px.line(fusion_df, x="date", y="FUSED_FEATURE"),
        use_container_width=True
    )

fig_fused = px.line(
    fusion_df,
    x="date",
    y="FUSED_FEATURE",
    title=f"Fused Feature: {fusion_x} + {fusion_y}"
)

st.plotly_chart(fig_fused, use_container_width=True)
