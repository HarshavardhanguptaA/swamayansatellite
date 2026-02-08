import numpy as np

# -----------------------------
# 1. Peak Lag Detection
# -----------------------------
def detect_peak_lag(lag_dict):
    if not lag_dict:
        return None, None

    best_lag = None
    best_corr = -np.inf

    for lag, corr in lag_dict.items():
        if corr is None or np.isnan(corr):
            continue
        if abs(corr) > best_corr:
            best_corr = abs(corr)
            best_lag = lag

    if best_lag is None:
        return None, None

    return best_lag, lag_dict[best_lag]



# -----------------------------
# 2. Stability Score
# -----------------------------
def stability_score(x, y):
    """
    Measures how stable the relationship is.
    Uses coefficient of variation of rolling correlation.
    """
    if len(x) < 6:
        return None

    window = max(3, len(x) // 4)
    rolling_corr = []

    for i in range(len(x) - window):
        r = np.corrcoef(x[i:i+window], y[i:i+window])[0, 1]
        if not np.isnan(r):
            rolling_corr.append(r)

    if len(rolling_corr) < 2:
        return None

    rolling_corr = np.array(rolling_corr)
    stability = np.std(rolling_corr) / (np.mean(np.abs(rolling_corr)) + 1e-6)

    return stability


# -----------------------------
# 3. Fusion Recommendation
# -----------------------------
def fusion_decision(corr, lag, stability):
    if corr is None or lag is None:
        return "Insufficient signal strength for fusion assessment."

    if stability is None:
        return "Relationship detected, but stability could not be reliably estimated."

    if abs(corr) < 0.3:
        return "Weak relationship detected. Fusion not recommended."

    if stability > 0.8:
        return "Relationship is unstable. Fusion may be unreliable."

    if lag == 0:
        return "Strong synchronous relationship. Feature-level fusion recommended."

    return (
        f"Strong relationship with a lag of {lag} timesteps. "
        "Lag-aware feature-level fusion recommended. "
        "Pixel-level fusion is not advised."
    )
