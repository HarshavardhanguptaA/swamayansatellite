import numpy as np

def lag_correlation(x, y, max_lag=8):
    """
    Robust lag correlation.
    Returns NaN where correlation cannot be computed.
    """

    results = {}

    n = min(len(x), len(y))

    for lag in range(-max_lag, max_lag + 1):

        if lag < 0:
            x_lag = x[:lag]
            y_lag = y[-lag:]

        elif lag > 0:
            x_lag = x[lag:]
            y_lag = y[:-lag]

        else:
            x_lag = x
            y_lag = y

        # 🛑 prevent empty or too-small arrays
        if len(x_lag) < 2 or len(y_lag) < 2:
            results[lag] = np.nan
            continue

        corr = np.corrcoef(x_lag, y_lag)[0, 1]
        results[lag] = corr

    return results
