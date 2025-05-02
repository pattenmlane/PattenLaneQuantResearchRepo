import pandas as pd
from sklearn.decomposition import PCA
RESAMPLE = '1min'
LVLS = range(1, 11)
def best_level_opi(events,lvl):
    i = lvl - 1
    bid_sz = f"bid_sz_{i:02d}"
    ask_sz = f"ask_sz_{i:02d}"
    d_bid = events[bid_sz].diff().clip(lower=0)
    d_ask = -events[ask_sz].diff().clip(upper=0)
    return d_bid.add(d_ask, fill_value=0)

def window_avg_depth(events, M=10):
    cols = [f"bid_sz_{i:02d}" for i in range(M)] + [f"ask_sz_{i:02d}" for i in range(M)]
    return events[cols].mean(axis=1)

def multi_level_ofi(sym_df):
    qbar_win = window_avg_depth(sym_df).resample(RESAMPLE).mean()
    out = {}
    for lvl in LVLS:
        raw_sum = best_level_ofi(sym_df,lvl).resample(RESAMPLE).sum(min_count=1)
        out[f"ofi_L{lvl}"] = raw_sum/ qbar_win
    return pd.DataFrame(out)

def integrated_ofi(multi_df):
    pca = PCA(n_components=1).fit(multi_df.fillna(0.0).values)
    w = pca.components_[0]
    w /= np.abs(w).sum()
    scores = (multi_df.fillna(0.0).values @ w).ravel()
    return pd.Series(scores, index=multi_df.index, name = "ofi_int")
    