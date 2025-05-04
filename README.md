# PattenLaneQuantResearchRepo
Order Flow Imbalance (OFI) Feature Generator
===========================================

This repository contains `ofi_features.py`, a self‑contained utility that extracts four flavours of Order‑Flow‑Imbalance (OFI) features on 1‑minute bars from event‑level limit‑order‑book data.  The implementation follows – and slightly extends – the methodology introduced in *Bormetti, Potters & Rindi (2023) “Cross‑Impact of Order Flow Imbalance in Equity Markets”.*

--------------------------------------------------------------------
1.  Background & Motivation
--------------------------------------------------------------------
The original paper shows that **signed order‑book pressure** (OFI) is a strong driver of short‑horizon price moves, and that *cross‑impact* — OFI in one asset predicting returns in another — is statistically significant.  Their key innovations are:

- Constructing OFI at **multiple depth levels**, not just at the best quotes.
- Using **Lasso regression** to estimate sparse cross‑impact coefficients in a large asset universe.

Our script reproduces the feature engineering part of that pipeline so the downstream modelling (OLS, Lasso, VAR, …) can be plugged in seamlessly.

--------------------------------------------------------------------
2.  Feature Definitions
--------------------------------------------------------------------

| Feature            | Formula                                                                  | Notes                                                                                         |
|--------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `ofi_best`         | `ΔBidSize − ΔAskSize` at the prevailing best prices                       | Equivalent to `ofi_lvl1`.                                                                     |
| `ofi_lvl1…lvlN`    | Same as above but computed **per depth level**; sizes sign‑adjusted when price moves | `NUM_LEVELS = 10` by default; automatically shrinks if the book is thinner.                   |
| `ofi_integrated`   | First principal‑component score of the *flow* matrix `[lvl1 … lvlN]`      | Weights learned *expanding‑window*; stored in `integrated_weights`.                           |
| `xofi_<peer>`      | `ofi_integrated` of every other symbol in the file                        | Provides ready‑made *cross‑asset predictors*.                                                 |

All flows are **normalised by average depth** (`(BidSize+AskSize)/2`) to account for intraday liquidity variation, yielding scale‑free features.

--------------------------------------------------------------------
3.  Implementation Choices & Deviations
--------------------------------------------------------------------
3.1  Bar Construction
    • Events are timestamped to UTC; we resample with `pd.Grouper(freq="1min", label="right", closed="right")`, meaning the bar ending at 10:01 contains trades with `ts_event ∈ (10:00, 10:01]`.
    • The very first event in the file is ignored for flow calculation to avoid spurious `prev_*` NaNs.

3.2  Depth Handling
    • `_pick_cols()` autoscans column names like `bid_px_00`, `bid_sz_00`, … up to ten levels, allowing partially‑sparse books.
    • `EPS_DEPTH = 1e‑9` guards against division by zero; extreme values can be clipped downstream.

3.3  Integrated OFI via PCA
    • **Expanding window** PCA (minimum 10 observations) avoids forward‑looking bias while letting the weights adapt intraday.
    • Component loadings are **L1‑normalised** so `ofi_integrated` is an easy‑to‑interpret weighted sum.
    • Weights are JSON‑serialised; `NaN` until the window is long enough.

3.4  Cross‑Asset OFI
    • After individual symbol aggregation, we pivot the integrated series wide and left‑join back, so each symbol row carries its peers’ integrated pressure.
    • This mimics the *predictor matrix* in the paper prior to their Lasso stage.

3.5  Parameter Grid

| Parameter       | Default | Rationale                                                |
|-----------------|---------|----------------------------------------------------------|
| `BAR_SECONDS`   | 60      | Matches most TAQ‑style academic work.                    |
| `NUM_LEVELS`    | 10      | Deeper than the paper’s 5 to stress‑test robustness.     |
| `MIN_OBS_PCA`   | 10      | Two trading hours at 1‑min bars before PCA kicks in.     |
| `EPS_DEPTH`     | 1e‑9    | Small enough to avoid bias; large enough to stop `inf`.  |

--------------------------------------------------------------------
4.  Usage
--------------------------------------------------------------------
```
# Install dependencies (Python ≥3.9)
pip install pandas numpy scikit-learn

# Generate feature file
python ofi_features.py path/to/raw.csv path/to/features.csv
```

*Input schema* (wide LOB format):
```
symbol, ts_event, bid_px_00, bid_sz_00, …, ask_px_00, ask_sz_00, …
```

*Output schema*:
```
symbol, ts_bar, ofi_best, ofi_lvl1 … lvl10, ofi_integrated, integrated_weights[, xofi_…]
```

--------------------------------------------------------------------
5.  Limitations & TODOs
--------------------------------------------------------------------
- **Extreme illiquidity**: when both sides are empty for a minute, depth normalisation can produce large numbers; consider post‑processing winsorisation.
- **Calendar effects**: the script assumes continuous trading; if the venue has auctions or halts you may wish to drop those intervals.
