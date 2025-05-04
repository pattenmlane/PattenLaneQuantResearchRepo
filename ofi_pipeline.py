"""
ofi_features.py
---------------
Spits out four OFI flavours per 1‑min bar:

• ofi_best           (best level)
• ofi_lvl1 … lvl10   (multi‑level)
• ofi_integrated     (PCA combo + its weights)
• xofi_<peer>        (peer integrated OFI, if >1 symbol)

Run:
    python ofi_features.py raw.csv  features.csv
"""

from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BAR_SECONDS   = 60
NUM_LEVELS    = 10
MIN_OBS_PCA   = 10
EPS_DEPTH     = 1e-9          # avoid ÷0

# ------------------------------------------------------------------
def _pick_cols(df: pd.DataFrame, side: str, what: str) -> List[str]:
    cols: List[str] = []
    for lvl in range(NUM_LEVELS):
        found = None
        for cand in (f"{side}_{what}_{lvl:02d}", f"{side}_{what}_{lvl}"):
            if cand in df.columns:
                found = cand
                break
        if found:                     # tolerate sparse books
            cols.append(found)
    if not cols:
        raise KeyError(f"No columns like {side}_{what}_00 found")
    return cols

# ------------------------------------------------------------------
def _signed_flows(c_px, p_px, c_sz, p_sz, side: str):
    if side == "bid":
        return np.where(c_px == p_px,
                        c_sz - p_sz,
                        np.where(c_px > p_px, c_sz, -p_sz))
    return np.where(c_px == p_px,
                    -(c_sz - p_sz),
                    np.where(c_px < p_px, -c_sz,  p_sz))

def _event_flows(df, BPX, BSZ, APX, ASZ):
    prev_bpx = df[BPX].shift().to_numpy(dtype=float)
    prev_bsz = df[BSZ].shift().to_numpy(dtype=float)
    prev_apx = df[APX].shift().to_numpy(dtype=float)
    prev_asz = df[ASZ].shift().to_numpy(dtype=float)

    cur_bpx  = df[BPX].to_numpy(dtype=float)
    cur_bsz  = df[BSZ].to_numpy(dtype=float)
    cur_apx  = df[APX].to_numpy(dtype=float)
    cur_asz  = df[ASZ].to_numpy(dtype=float)

    prev_bpx[0], prev_bsz[0] = cur_bpx[0], cur_bsz[0]
    prev_apx[0], prev_asz[0] = cur_apx[0], cur_asz[0]

    bid = _signed_flows(cur_bpx, prev_bpx, cur_bsz, prev_bsz, "bid")
    ask = _signed_flows(cur_apx, prev_apx, cur_asz, prev_asz, "ask")
    return bid + ask                                # n_events × n_levels

# ------------------------------------------------------------------
def _symbol_ofi(df_sym: pd.DataFrame) -> pd.DataFrame:
    df_sym = df_sym.copy()
    df_sym["ts_event"] = pd.to_datetime(df_sym["ts_event"], utc=True)
    df_sym = df_sym.sort_values("ts_event").reset_index(drop=True)

    BPX = _pick_cols(df_sym, "bid", "px")
    BSZ = _pick_cols(df_sym, "bid", "sz")
    APX = _pick_cols(df_sym, "ask", "px")
    ASZ = _pick_cols(df_sym, "ask", "sz")
    n_lvls = min(len(BPX), len(APX), NUM_LEVELS)     # handle sparse depth

    flows = _event_flows(df_sym, BPX[:n_lvls], BSZ[:n_lvls],
                                   APX[:n_lvls], ASZ[:n_lvls])
    flow_cols  = [f"flow_lvl{i+1}"  for i in range(n_lvls)]
    depth_cols = [f"depth_lvl{i+1}" for i in range(n_lvls)]
    df_sym[flow_cols] = flows

    for i in range(n_lvls):
        df_sym[depth_cols[i]] = (
            df_sym[BSZ[i]] + df_sym[ASZ[i]]
        ).astype(float) / 2.0

    res = (
        df_sym.set_index("ts_event")
        .groupby(pd.Grouper(freq="1min", label="right"))
        .agg({**{c: "sum"  for c in flow_cols},
              **{c: "mean" for c in depth_cols}})
        .dropna(subset=[flow_cols[0]])
    )

    for i in range(n_lvls):
        res[flow_cols[i]] /= (res[depth_cols[i]] + EPS_DEPTH)

    ofi_cols = [f"ofi_lvl{i+1}" for i in range(n_lvls)]
    res = res.rename(columns=dict(zip(flow_cols, ofi_cols)))
    res["ofi_best"] = res["ofi_lvl1"]

    # expanding PCA
    ints, wts = [], []
    pca = PCA(n_components=1)
    for i in range(len(res)):
        win = res.iloc[: i + 1][ofi_cols]
        if len(win) < MIN_OBS_PCA:
            ints.append(np.nan)
            wts.append([np.nan] * n_lvls)
            continue
        pca.fit(win.values)
        w = pca.components_[0]
        w /= np.abs(w).sum()
        ints.append(float(np.dot(w, res.iloc[i][ofi_cols])))
        wts.append(w.tolist())

    res["ofi_integrated"]     = ints
    res["integrated_weights"] = [json.dumps(x) for x in wts]
    keep = ["ofi_best", *ofi_cols, "ofi_integrated", "integrated_weights"]
    res = res[keep]
    res.index.name = "ts_bar"
    return res

# ------------------------------------------------------------------
def _add_cross(ofi_by_sym: Dict[str, pd.DataFrame]):
    if len(ofi_by_sym) < 2:
        return ofi_by_sym
    wide = pd.concat({s: df["ofi_integrated"] for s, df in ofi_by_sym.items()},
                     axis=1).sort_index()
    for s, df in ofi_by_sym.items():
        peers = [p for p in wide.columns if p != s]
        ofi_by_sym[s] = df.join(
            wide[peers].rename(columns={p: f"xofi_{p}" for p in peers}),
            how="left"
        )
    return ofi_by_sym

# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python ofi_features.py input.csv output.csv")

    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    raw = pd.read_csv(in_path)
    if "symbol" not in raw.columns:
        raw["symbol"] = "UNKNOWN"

    ofi = {s: _symbol_ofi(g) for s, g in raw.groupby("symbol", sort=False)}
    ofi = _add_cross(ofi)

    out = (pd.concat(ofi, names=["symbol", "ts_bar"])
           .reset_index()
           .sort_values(["symbol", "ts_bar"]))
    out.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()


