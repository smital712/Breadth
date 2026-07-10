"""
NSE P&F Bull Breadth Dashboard — Streamlit App
===================================================
Self-contained: all P&F / breadth / NSE-constituent logic included.
No separate imports from b5WorkingXOBreadth.py required.

Run locally:
    pip install -r examples/python/requirements.txt
    streamlit run examples/python/streamlit_breadth.py

Deploy on Streamlit Community Cloud (free):
    1. Push this repo to GitHub (public repo required for free tier)
    2. Go to https://share.streamlit.io  →  New app
    3. Repository: <your-github-repo>  |  Branch: main
       Main file path:  examples/python/streamlit_breadth.py
    4. Advanced settings  →  Python packages:
         copy requirements.txt contents to repo root, OR point to
         examples/python/requirements.txt in the Packages field
    5. Click Deploy — first run fetches from Yahoo Finance & NSE (free/public)

Performance:
    • Nifty 50  (~50 symbols)  →  ~30 sec first run, instant on re-run (cache)
    • Nifty MidSmallCap 400    →  5-15 min first run
    • Changing only the SMA slider never re-fetches data
"""
from __future__ import annotations

import calendar
import io
import json
import math
import logging
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ═══════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════
LOGGER = logging.getLogger("pf_breadth_app")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════
#  LAYER 0 – SHARED UTILS
# ═══════════════════════════════════════════════════════════════

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if str(n).strip().lower() in cols:
            return cols[str(n).strip().lower()]
    return None


def ensure_canonical_ohlc(df_any: pd.DataFrame) -> pd.DataFrame:
    if df_any is None or df_any.empty:
        return pd.DataFrame(columns=["DateTime", "Open", "High", "Low", "Close", "Volume"])
    df = _lower_cols(df_any)
    ts_col = _pick_col(df, "timestamp", "datetime", "date")
    if ts_col is None:
        dt = df_any.index if isinstance(df_any.index, pd.DatetimeIndex) else pd.to_datetime(df_any.index, errors="coerce")
    else:
        dt = pd.to_datetime(df[ts_col], errors="coerce")
    o = _pick_col(df, "open", "o")
    h = _pick_col(df, "high", "h")
    l = _pick_col(df, "low", "l")
    c = _pick_col(df, "close", "c")
    v = _pick_col(df, "volume", "vol", "v")
    missing = [k for k, col in [("open", o), ("high", h), ("low", l), ("close", c)] if col is None]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Have: {list(df_any.columns)}")
    out = pd.DataFrame({
        "DateTime": dt,
        "Open":  pd.to_numeric(df[o], errors="coerce"),
        "High":  pd.to_numeric(df[h], errors="coerce"),
        "Low":   pd.to_numeric(df[l], errors="coerce"),
        "Close": pd.to_numeric(df[c], errors="coerce"),
    })
    if v is not None:
        out["Volume"] = pd.to_numeric(df[v], errors="coerce")
    out = out.dropna(subset=["DateTime", "Close"]).sort_values("DateTime").reset_index(drop=True)
    if isinstance(out["DateTime"].dtype, pd.DatetimeTZDtype):
        out["DateTime"] = out["DateTime"].dt.tz_localize(None)
    return out


def is_intraday_df(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    dt = df["DateTime"]
    return ((dt.dt.hour != 0) | (dt.dt.minute != 0) | (dt.dt.second != 0)).any()


def _norm_type(x: Any) -> str:
    return str(x).strip().lower()


def _copy_or_inplace(df: pd.DataFrame, inplace: bool) -> pd.DataFrame:
    return df if inplace else df.copy()


def _missing(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c not in df.columns]


def _ensure_cols(df: pd.DataFrame, cols: Sequence[str], *, where: str, logger: logging.Logger) -> None:
    miss = _missing(df, cols)
    if miss:
        raise ValueError(f"{where}: missing required columns: {miss}")


_INTERNAL_COLS = {"_DTB_Level", "_DBS_Level"}


def _maybe_drop_internal(df: pd.DataFrame, *, keep_internal: bool) -> pd.DataFrame:
    if keep_internal:
        return df
    return df.drop(columns=[c for c in _INTERNAL_COLS if c in df.columns], errors="ignore")


# ═══════════════════════════════════════════════════════════════
#  LAYER 1 – P&F ENGINE  (subset needed for breadth)
# ═══════════════════════════════════════════════════════════════

def build_grid_helpers(pct: float, round_dp: int) -> Dict[str, Any]:
    r = 1.0 + pct / 100.0
    log_r = math.log(r)
    eps = 1e-12

    def price_from_k(k: int) -> float:
        return round(math.exp(float(k) * log_r), round_dp)

    def k_from_price(price: float) -> int:
        raw = math.log(float(price)) / log_r
        k0 = int(round(raw))
        bestk, besterr = k0, float("inf")
        for kk in range(k0 - 4, k0 + 5):
            err = abs(price_from_k(kk) - float(price))
            if err < besterr:
                besterr, bestk = err, kk
        return bestk

    def next_box(price: float) -> float:
        return price_from_k(k_from_price(price) + 1)

    def prev_box(price: float) -> float:
        return price_from_k(k_from_price(price) - 1)

    def box_at_or_below(price: float) -> float:
        k = k_from_price(price)
        p = price_from_k(k)
        return price_from_k(k - 1) if p > float(price) + eps else p

    def box_at_or_above(price: float) -> float:
        k = k_from_price(price)
        p = price_from_k(k)
        return price_from_k(k + 1) if p < float(price) - eps else p

    return {
        "r": r, "price_from_k": price_from_k, "k_from_price": k_from_price,
        "next_box": next_box, "prev_box": prev_box,
        "box_at_or_below": box_at_or_below, "box_at_or_above": box_at_or_above,
    }


def daily_first_move_boxes_default(pct: float) -> int:
    return 1 if abs(pct - 1.0) < 1e-12 else 3


def generate_pnf_columns_algo(
    df: pd.DataFrame,
    pct: float,
    reversal_boxes: int = 3,
    round_dp: int = 2,
    definedge_first_column: bool = True,
    first_move_boxes: Optional[int] = None,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=[
        "Column", "Type", "Start_Date", "Date", "Close",
        "High", "High_Date", "Low", "Low_Date",
        "Box_Count", "Mid_Value", "Reversal_Value", "Continuation_Value",
        "_High_K", "_Low_K",
    ])
    if df is None or df.empty:
        return empty

    work = df[["DateTime", "Close"]].copy()
    work = work.dropna(subset=["DateTime", "Close"]).sort_values("DateTime").reset_index(drop=True)
    if work.empty:
        return empty

    intraday = is_intraday_df(work)
    grid = build_grid_helpers(pct, round_dp)
    log_r = math.log(grid["r"])
    eps = 1e-12

    def k_floor(price):
        return int(math.floor(math.log(float(price)) / log_r + eps))

    def k_ceil(price):
        return int(math.ceil(math.log(float(price)) / log_r - eps))

    def price_from_k(k):
        return grid["price_from_k"](k)

    def dt_str(i):
        fmt = "%Y-%m-%d %H:%M:%S" if intraday else "%Y-%m-%d"
        return work.loc[i, "DateTime"].strftime(fmt)

    c0 = float(work.loc[0, "Close"])
    anchor_k = k_floor(c0) if (pct < 0.05 or abs(pct - 1.0) < 1e-12 or pct > 1.0) else k_ceil(c0)
    first_move = daily_first_move_boxes_default(pct) if first_move_boxes is None else int(first_move_boxes)

    cols: List[dict] = []
    direction: Optional[str] = None

    for i in range(1, len(work)):
        close = float(work.loc[i, "Close"])
        up_k = k_floor(close)
        dn_k = k_ceil(close)
        dn_floor = k_floor(close)

        if direction is None:
            if up_k >= anchor_k + first_move:
                cols.append({"type": "x", "hi_k": up_k, "lo_k": anchor_k,
                              "hi_i": i, "lo_i": 0, "start_i": 0})
                direction = "x"
            else:
                down_start_k = dn_floor if definedge_first_column else dn_k
                if down_start_k <= anchor_k - first_move:
                    cols.append({"type": "o", "hi_k": anchor_k, "lo_k": dn_floor,
                                  "hi_i": 0, "lo_i": i, "start_i": 0})
                    direction = "o"
            continue

        cur = cols[-1]
        if direction == "x":
            if up_k > cur["hi_k"]:
                cur["hi_k"] = up_k
                cur["hi_i"] = i
            if dn_k <= cur["hi_k"] - reversal_boxes:
                cols.append({"type": "o", "hi_k": cur["hi_k"] - 1, "lo_k": dn_k,
                              "hi_i": i, "lo_i": i, "start_i": i})
                direction = "o"
        else:
            if dn_k < cur["lo_k"]:
                cur["lo_k"] = dn_k
                cur["lo_i"] = i
            if up_k >= cur["lo_k"] + reversal_boxes:
                cols.append({"type": "x", "hi_k": up_k, "lo_k": cur["lo_k"] + 1,
                              "hi_i": i, "lo_i": i, "start_i": i})
                direction = "x"

    out_rows = []
    for idx, c in enumerate(cols, start=1):
        hi_k, lo_k = int(c["hi_k"]), int(c["lo_k"])
        hi, lo = price_from_k(hi_k), price_from_k(lo_k)
        box_count = hi_k - lo_k + 1
        mid_val = round((hi + lo) / 2.0, round_dp)
        if c["type"] == "x":
            reversal_val  = price_from_k(hi_k - reversal_boxes)
            continuation_val = price_from_k(hi_k + 1)
        else:
            reversal_val  = price_from_k(lo_k + reversal_boxes)
            continuation_val = price_from_k(lo_k - 1)
        out_rows.append({
            "Column": idx, "Type": c["type"],
            "Start_Date": dt_str(c["start_i"]),
            "Date": dt_str(c["hi_i"]) if c["type"] == "x" else dt_str(c["lo_i"]),
            "Close": hi if c["type"] == "x" else lo,
            "High": hi, "High_Date": dt_str(c["hi_i"]),
            "Low":  lo, "Low_Date":  dt_str(c["lo_i"]),
            "Box_Count": box_count, "Mid_Value": mid_val,
            "Reversal_Value": reversal_val, "Continuation_Value": continuation_val,
            "_High_K": hi_k, "_Low_K": lo_k,
        })
    return pd.DataFrame(out_rows)


@dataclass(frozen=True)
class PatternConfig:
    pct: float
    round_dp: int = 2
    strict: bool = True
    allow_fallback_k: bool = True
    k_validate_tol_boxes: int = 0
    logger: logging.Logger = LOGGER

    @property
    def grid(self) -> Dict[str, Any]:
        return build_grid_helpers(self.pct, self.round_dp)


def compute_rolling_xo_zone(types: List[str], box_counts: List[int], window: int = 10):
    n = len(types)
    xout, oout, zout, zpctout = [0]*n, [0]*n, [0]*n, [0.0]*n
    for i in range(n):
        x = o = 0
        for j in range(max(0, i - window + 1), i + 1):
            bc = int(box_counts[j]) if j < len(box_counts) else 0
            if types[j] == "x":
                x += bc
            elif types[j] == "o":
                o += bc
        z = x - o
        denom = x + o
        xout[i], oout[i], zout[i] = x, o, z
        zpctout[i] = round(z * 100.0 / denom, 2) if denom > 0 else 0.0
    return xout, oout, zout, zpctout


def pf_xo_zone(pf, *, window=10, inplace=False, cfg=None, keep_internal=False):
    cfg = cfg or PatternConfig(pct=1.0)
    out = _copy_or_inplace(pf, inplace)
    _ensure_cols(out, ["Type", "Box_Count"], where="pf_xo_zone", logger=cfg.logger)
    types      = [_norm_type(x) for x in out["Type"].tolist()]
    box_counts = [int(x) if pd.notna(x) else 0 for x in out["Box_Count"].tolist()]
    x_cnt, o_cnt, z, z_pct = compute_rolling_xo_zone(types, box_counts, window=window)
    out["X Count"] = x_cnt
    out["O Count"] = o_cnt
    out["XO Zone"]   = z
    out["XO Zone %"] = z_pct
    return _maybe_drop_internal(out, keep_internal=keep_internal)


# ═══════════════════════════════════════════════════════════════
#  LAYER 2 – STANDARD BREADTH  (per-symbol helper)
# ═══════════════════════════════════════════════════════════════

def fetch_symbol_ohlc_yahoo(
    symbol: str,
    exchange: str = "NSE",
    interval: str = "D",
    startdate: str = "2020-01-01",
    enddate: Optional[str] = None,
) -> pd.DataFrame:
    if enddate is None:
        enddate = datetime.now().strftime("%Y-%m-%d")
    interval_map = {
        "D": "1d", "1d": "1d", "W": "1wk", "1wk": "1wk",
        "M": "1mo", "1mo": "1mo", "60": "60m", "15": "15m", "5": "5m", "1": "1m",
    }
    yf_interval = interval_map.get(interval, "1d")
    if exchange.upper() == "NSE" and not symbol.upper().endswith(".NS"):
        ticker = f"{symbol}.NS"
    elif exchange.upper() == "BSE" and not symbol.upper().endswith(".BO"):
        ticker = f"{symbol}.BO"
    else:
        ticker = symbol
    df = yf.download(tickers=ticker, start=startdate, end=enddate,
                     interval=yf_interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    else:
        df.columns = [str(c) for c in df.columns]
    df = df.reset_index().rename(columns={"Date": "DateTime"})
    return ensure_canonical_ohlc(df)


def symbol_xozone_on_ohlc_dates(
    df_ohlc: pd.DataFrame,
    pct: float = 1.0,
    reversal_boxes: int = 3,
    rounddp: int = 2,
    xozone_window: int = 10,
) -> pd.DataFrame:
    if df_ohlc is None or df_ohlc.empty:
        return pd.DataFrame(columns=["Date", "XOZone", "XOZoneFlag"])
    o = df_ohlc.copy()
    o["Date"] = pd.to_datetime(o["DateTime"], errors="coerce").dt.normalize()
    cal = pd.DataFrame({"Date": sorted(o["Date"].dropna().unique())})
    if cal.empty:
        return pd.DataFrame(columns=["Date", "XOZone", "XOZoneFlag"])
    pf = generate_pnf_columns_algo(df_ohlc, pct=pct, reversal_boxes=reversal_boxes, round_dp=rounddp)
    cfg = PatternConfig(pct=pct, round_dp=rounddp, strict=True)
    pf2 = pf_xo_zone(pf, window=xozone_window, cfg=cfg, inplace=False, keep_internal=False)
    if pf2.empty or "Start_Date" not in pf2.columns or "XO Zone" not in pf2.columns:
        out = cal.copy()
        out["XOZone"]    = np.nan
        out["XOZoneFlag"] = 0
        return out
    p = pf2[["Start_Date", "XO Zone"]].copy()
    p["Start_Date"] = pd.to_datetime(p["Start_Date"], errors="coerce").dt.normalize()
    p = p.dropna(subset=["Start_Date"]).sort_values("Start_Date")
    p["XOZone"] = pd.to_numeric(p["XO Zone"], errors="coerce")
    p = p.rename(columns={"Start_Date": "Date"})[["Date", "XOZone"]].sort_values("Date")
    mapped = pd.merge_asof(
        cal.sort_values("Date"), p.sort_values("Date"),
        on="Date", direction="backward", allow_exact_matches=True,
    )
    mapped["XOZone"]    = mapped["XOZone"].ffill()
    mapped["XOZoneFlag"] = (mapped["XOZone"] > 0).fillna(False).astype(int)
    return mapped[["Date", "XOZone", "XOZoneFlag"]]


# ═══════════════════════════════════════════════════════════════
#  LAYER 2B – DAILY SCANNER BREADTH  (per-symbol helper)
# ═══════════════════════════════════════════════════════════════

@dataclass
class _PnfScanState:
    """Mutable incremental P&F state for the daily scanner.  One instance per symbol."""
    anchor_k:   int
    first_move: int
    direction:  Optional[str] = None
    cur_type:   Optional[str] = None
    cur_hi_k:   Optional[int] = None
    cur_lo_k:   Optional[int] = None
    completed_cols: List[Dict[str, Any]] = field(default_factory=list)


def _pnf_scan_update(
    state: _PnfScanState,
    close: float,
    k_floor,
    k_ceil,
    reversal_boxes: int,
) -> None:
    up_k    = k_floor(close)
    dn_k    = k_ceil(close)
    dn_floor = k_floor(close)

    if state.direction is None:
        if up_k >= state.anchor_k + state.first_move:
            state.direction = "x"
            state.cur_type  = "x"
            state.cur_hi_k  = up_k
            state.cur_lo_k  = state.anchor_k
        elif dn_floor <= state.anchor_k - state.first_move:
            state.direction = "o"
            state.cur_type  = "o"
            state.cur_hi_k  = state.anchor_k
            state.cur_lo_k  = dn_floor
        return

    if state.direction == "x":
        if up_k > state.cur_hi_k:
            state.cur_hi_k = up_k
        if dn_k <= state.cur_hi_k - reversal_boxes:
            state.completed_cols.append({
                "type": "x",
                "box_count": state.cur_hi_k - state.cur_lo_k + 1,
            })
            new_hi_k = state.cur_hi_k - 1
            state.direction = "o"
            state.cur_type  = "o"
            state.cur_hi_k  = new_hi_k
            state.cur_lo_k  = dn_k
    else:
        if dn_k < state.cur_lo_k:
            state.cur_lo_k = dn_k
        if up_k >= state.cur_lo_k + reversal_boxes:
            state.completed_cols.append({
                "type": "o",
                "box_count": state.cur_hi_k - state.cur_lo_k + 1,
            })
            new_lo_k = state.cur_lo_k + 1
            state.direction = "x"
            state.cur_type  = "x"
            state.cur_lo_k  = new_lo_k
            state.cur_hi_k  = up_k


def _xo_zone_from_scan_state(state: _PnfScanState, xozone_window: int) -> Tuple[float, int]:
    if state.direction is None or state.cur_type is None:
        return 0.0, 0
    cur_bc = max(state.cur_hi_k - state.cur_lo_k + 1, 1)
    all_cols: List[Dict[str, Any]] = state.completed_cols + [
        {"type": state.cur_type, "box_count": cur_bc}
    ]
    window_cols = all_cols[-xozone_window:]
    x_sum = sum(c["box_count"] for c in window_cols if c["type"] == "x")
    o_sum = sum(c["box_count"] for c in window_cols if c["type"] == "o")
    xo_zone = float(x_sum - o_sum)
    return xo_zone, (1 if xo_zone > 0 else 0)


def symbol_xozone_daily_scanner(
    df_ohlc: pd.DataFrame,
    pct: float = 1.0,
    reversal_boxes: int = 3,
    rounddp: int = 2,
    xozone_window: int = 10,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["Date", "XOZone", "XOZoneFlag"])
    if df_ohlc is None or df_ohlc.empty:
        return empty

    df = ensure_canonical_ohlc(df_ohlc)
    if df.empty or len(df) < 2:
        return empty

    grid  = build_grid_helpers(pct, rounddp)
    log_r = math.log(grid["r"])
    eps   = 1e-12

    def k_floor(price: float) -> int:
        return int(math.floor(math.log(float(price)) / log_r + eps))

    def k_ceil(price: float) -> int:
        return int(math.ceil(math.log(float(price)) / log_r - eps))

    c0 = float(df.loc[0, "Close"])
    anchor_k = k_floor(c0) if (pct < 0.05 or abs(pct - 1.0) < 1e-12 or pct > 1.0) else k_ceil(c0)
    first_move = daily_first_move_boxes_default(pct)

    state    = _PnfScanState(anchor_k=anchor_k, first_move=first_move)
    intraday = is_intraday_df(df)
    rows: List[Dict[str, Any]] = []

    date_0 = pd.Timestamp(df.loc[0, "DateTime"])
    rows.append({"Date": date_0 if intraday else date_0.normalize(), "XOZone": 0.0, "XOZoneFlag": 0})

    for i in range(1, len(df)):
        close = float(df.loc[i, "Close"])
        _pnf_scan_update(state, close, k_floor, k_ceil, reversal_boxes)
        xo_val, xo_flag = _xo_zone_from_scan_state(state, xozone_window)
        date_i = pd.Timestamp(df.loc[i, "DateTime"])
        rows.append({"Date": date_i if intraday else date_i.normalize(),
                     "XOZone": xo_val, "XOZoneFlag": xo_flag})

    return pd.DataFrame(rows, columns=["Date", "XOZone", "XOZoneFlag"])


# ═══════════════════════════════════════════════════════════════
#  NSE INDEX CONSTITUENT HELPERS
# ═══════════════════════════════════════════════════════════════

_NSE_INDEX_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty{key}list.csv"
_NSE_INDEX_ZIP_URL = (
    "https://www.niftyindices.com/Indices_-_Market_Capitalisation_and_Weightage/"
    "indices_data{mon}{yyyy}.zip"
)
_NSE_SHEET_NAMES: Dict[str, str] = {
    "midsmallcap400": "NIFTY MIDSMALLCAP 400",
    "500":            "NIFTY 500",
    "50":             "NIFTY 50",
    "midcap150":      "NIFTY MIDCAP 150",
    "smallcap250":    "NIFTY SMALLCAP 250",
    "100":            "NIFTY 100",
    "200":            "NIFTY 200",
    "next50":         "NIFTY NEXT 50",
}
_NSE_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.niftyindices.com/",
}


def _nse_http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=_NSE_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_nse_index_constituents(
    index_key: str = "midsmallcap400",
    cache_file: Optional[str] = None,
    cache_ttl_hours: float = 24,
) -> List[str]:
    if cache_file is None:
        os.makedirs("data", exist_ok=True)
        cache_file = os.path.join("data", f"nse_{index_key}_current.json")

    if os.path.exists(cache_file):
        age_h = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < cache_ttl_hours:
            with open(cache_file, encoding="utf-8") as fh:
                return json.load(fh)

    url = _NSE_INDEX_CSV_URL.format(key=index_key)
    try:
        raw = _nse_http_get(url)
        df  = pd.read_csv(io.BytesIO(raw))
        sym_col = next(c for c in df.columns if c.strip().lower() == "symbol")
        syms = [s for s in df[sym_col].str.strip().tolist() if isinstance(s, str) and s]
        if not syms:
            raise ValueError("Parsed symbol list is empty")
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(syms, fh)
        return syms
    except Exception as exc:
        if os.path.exists(cache_file):
            with open(cache_file, encoding="utf-8") as fh:
                return json.load(fh)
        raise RuntimeError(
            f"Cannot fetch NSE constituents for '{index_key}' — no cache at '{cache_file}'."
        ) from exc


def fetch_historical_nse_constituents(
    index_key: str,
    year: int,
    month: int,
    cache_dir: str = "data",
) -> List[str]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"nse_{index_key}_{year}_{month:02d}.json")

    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as fh:
            return json.load(fh)

    mon_str = calendar.month_abbr[month]
    url     = _NSE_INDEX_ZIP_URL.format(mon=mon_str, yyyy=year)
    raw_zip = _nse_http_get(url)
    sheet_name = _NSE_SHEET_NAMES.get(index_key, index_key.upper())

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))]
        if not xlsx_names:
            raise FileNotFoundError(f"No Excel file found in ZIP from: {url}")
        xlsx_bytes = zf.read(xlsx_names[0])

    xl      = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
    target  = sheet_name.lower()
    matched = next(
        (s for s in xl.sheet_names if target in s.lower() or s.lower() in target), None
    )
    if matched is None:
        raise KeyError(
            f"Sheet '{sheet_name}' not found for {index_key} {year}-{month:02d}. "
            f"Available: {', '.join(xl.sheet_names)}"
        )

    df      = xl.parse(matched)
    sym_col = next(c for c in df.columns if c.strip().lower() == "symbol")
    syms    = [s for s in df[sym_col].str.strip().tolist() if isinstance(s, str) and s]
    with open(cache_file, "w", encoding="utf-8") as fh:
        json.dump(syms, fh)
    return syms


def get_constituents(
    index_key: str = "midsmallcap400",
    as_of_date: Optional[str] = None,
    cache_ttl_hours: float = 24,
) -> List[str]:
    if as_of_date is None:
        return fetch_nse_index_constituents(index_key, cache_ttl_hours=cache_ttl_hours)
    ts = pd.Timestamp(as_of_date)
    return fetch_historical_nse_constituents(index_key, year=ts.year, month=ts.month)


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT CACHE WRAPPERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _cached_fetch(
    symbol: str, exchange: str, interval: str, startdate: str, enddate: str
) -> pd.DataFrame:
    """Cache Yahoo Finance OHLC per (symbol, date-range) for 1 hour."""
    return fetch_symbol_ohlc_yahoo(
        symbol, exchange=exchange, interval=interval,
        startdate=startdate, enddate=enddate,
    )


@st.cache_data(ttl=3600 * 24, show_spinner=False)
def _cached_constituents(index_key: str, as_of_date: Optional[str]) -> List[str]:
    """Cache NSE constituent list for 24 hours, with fallback to current list."""
    try:
        return get_constituents(index_key, as_of_date=as_of_date)
    except Exception:
        return fetch_nse_index_constituents(index_key)


# ═══════════════════════════════════════════════════════════════
#  CHART BUILDER
# ═══════════════════════════════════════════════════════════════

def build_chart(
    breadth_df: pd.DataFrame,
    scanner_df: pd.DataFrame,
    sma_period: int,
    title: str,
) -> go.Figure:
    """Return Plotly figure: scanner breadth + SMA + standard breadth + reference lines."""
    fig = go.Figure()

    sma = scanner_df["BreadthPct"].rolling(sma_period, min_periods=1).mean().round(2)

    # Trace 1 — Standard breadth (Layer 2): medium grey solid
    if not breadth_df.empty:
        fig.add_trace(go.Scatter(
            x=breadth_df["Date"],
            y=breadth_df["BreadthPct"].round(2),
            name="Standard Breadth (Layer 2)",
            mode="lines",
            line=dict(color="#888888", width=2, dash="dash"),
            hovertemplate="<b>Std</b>: %{y:.1f}%<extra></extra>",
        ))

    # Trace 2 — Scanner breadth (Layer 2B): vivid blue solid
    fig.add_trace(go.Scatter(
        x=scanner_df["Date"],
        y=scanner_df["BreadthPct"].round(2),
        name="Scanner Breadth (Layer 2B)",
        mode="lines",
        line=dict(color="#1565C0", width=2.5),
        customdata=np.stack(
            [scanner_df["UpCount"].values, scanner_df["TotalCount"].values], axis=1
        ),
        hovertemplate=(
            "<b>Scanner</b>: %{y:.1f}%<br>"
            "Up: %{customdata[0]} / %{customdata[1]}<extra></extra>"
        ),
    ))

    # Trace 3 — SMA of scanner: vivid orange solid
    fig.add_trace(go.Scatter(
        x=scanner_df["Date"],
        y=sma,
        name=f"{sma_period}-SMA of Scanner",
        mode="lines",
        line=dict(color="#E65100", width=3),
        hovertemplate=f"<b>{sma_period}-SMA</b>: %{{y:.1f}}%<extra></extra>",
    ))

    # Reference lines
    _refs = [
        (70, "#2E7D32", "70% — overbought"),
        (50, "#616161", "50% — midpoint"),
        (30, "#C62828", "30% — oversold"),
    ]
    for level, color, label in _refs:
        fig.add_hline(
            y=level,
            line=dict(color=color, width=1.5, dash="dot"),
            annotation_text=label,
            annotation_position="top left",
            annotation_font=dict(size=10, color=color),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True, thickness=0.05),
            type="date",
            gridcolor="#E0E0E0",
        ),
        yaxis=dict(
            title="Breadth %",
            range=[0, 100],
            ticksuffix="%",
            gridcolor="#E0E0E0",
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=530,
        template="plotly_white",
        plot_bgcolor="#FAFAFA",
        margin=dict(l=60, r=20, t=80, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════

_INDEX_OPTIONS: Dict[str, str] = {
    "50":             "Nifty 50  (~50 symbols, ~30 sec)",
    "next50":         "Nifty Next 50  (~50 symbols)",
    "100":            "Nifty 100  (~100 symbols)",
    "200":            "Nifty 200  (~200 symbols)",
    "midcap150":      "Nifty Midcap 150",
    "smallcap250":    "Nifty Smallcap 250",
    "midsmallcap400": "Nifty MidSmallCap 400  (~400 symbols, 5-15 min)",
    "500":            "Nifty 500  (~500 symbols, very slow)",
}
_DEFAULT_INDEX = "50"


def _short_label(key: str) -> str:
    return _INDEX_OPTIONS[key].split("(")[0].strip()


def main() -> None:
    st.set_page_config(
        page_title="NSE P&F Bull Breadth",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ─── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Breadth Dashboard")
        st.caption("NSE P&F Bull Breadth — Layer 2 vs Layer 2B")
        st.divider()

        st.subheader("Universe")
        index_key = st.selectbox(
            "Index",
            options=list(_INDEX_OPTIONS.keys()),
            format_func=lambda k: _INDEX_OPTIONS[k],
            index=list(_INDEX_OPTIONS.keys()).index(_DEFAULT_INDEX),
        )

        st.subheader("Date Range")
        today = pd.Timestamp.now().normalize()
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input(
                "Start", value=(today - pd.Timedelta(days=365)).date()
            )
        with col_b:
            end_date = st.date_input("End", value=today.date())

        if start_date >= end_date:
            st.error("Start must be before End.")

        pct, reversal_boxes, xozone_window = 1.0, 3, 10  # fixed P&F defaults

        sma_period = st.slider(
            "SMA Period", min_value=1, max_value=50, value=5, step=1
        )

        st.divider()
        run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        if "run_params" in st.session_state:
            p = st.session_state.run_params
            st.caption(
                f"Last: {_short_label(p['index_key'])}  |  "
                f"{p['start_date']} → {p['end_date']}"
            )

    # ─── Main area ─────────────────────────────────────────────
    st.title("NSE P&F Bull Breadth Dashboard")

    if run_clicked:
        if start_date >= end_date:
            st.error("Fix the date range first.")
            return

        # Load symbol universe
        with st.spinner("Loading symbol universe from NSE…"):
            try:
                symbols = _cached_constituents(index_key, as_of_date=str(start_date))
            except Exception as exc:
                st.error(f"Failed to load symbol list: {exc}")
                st.info("NSE website may be temporarily unavailable — try again.")
                return

        n_symbols   = len(symbols)
        fetch_start = (
            pd.Timestamp(str(start_date)) - pd.Timedelta(days=1825)
        ).strftime("%Y-%m-%d")
        output_from = pd.Timestamp(str(start_date)).normalize()

        st.info(
            f"Processing **{n_symbols} symbols** for {_short_label(index_key)}  |  "
            f"{start_date} → {end_date}"
        )

        # Per-symbol loop with progress bar
        pbar    = st.progress(0.0)
        status  = st.empty()

        xoz_wide  = flag_wide  = None   # Layer 2
        s_xoz_wide = s_flag_wide = None  # Layer 2B

        for i, sym in enumerate(symbols):
            pbar.progress((i + 1) / n_symbols, text=f"{sym}  ({i+1}/{n_symbols})")

            try:
                df = _cached_fetch(sym, "NSE", "D", fetch_start, str(end_date))
                if df.empty:
                    continue

                # ── Layer 2: standard breadth (P&F column start-date mapped) ──
                daily = symbol_xozone_on_ohlc_dates(
                    df, pct=pct, reversal_boxes=reversal_boxes,
                    rounddp=2, xozone_window=xozone_window,
                )
                if not daily.empty:
                    daily = daily[daily["Date"] >= output_from]
                    if not daily.empty:
                        xo_col = daily.set_index("Date")["XOZone"].rename(sym)
                        fl_col = daily.set_index("Date")["XOZoneFlag"].rename(sym)
                        xoz_wide  = (xo_col.to_frame() if xoz_wide  is None else xoz_wide.join(xo_col,  how="outer"))
                        flag_wide = (fl_col.to_frame() if flag_wide is None else flag_wide.join(fl_col, how="outer"))

                # ── Layer 2B: daily scanner breadth (row-by-row P&F state) ──
                scanner_daily = symbol_xozone_daily_scanner(
                    df, pct=pct, reversal_boxes=reversal_boxes,
                    rounddp=2, xozone_window=xozone_window,
                )
                if not scanner_daily.empty:
                    scanner_daily = scanner_daily[scanner_daily["Date"] >= output_from]
                    if not scanner_daily.empty:
                        s_xo = scanner_daily.set_index("Date")["XOZone"].rename(sym)
                        s_fl = scanner_daily.set_index("Date")["XOZoneFlag"].rename(sym)
                        s_xoz_wide  = (s_xo.to_frame() if s_xoz_wide  is None else s_xoz_wide.join(s_xo,  how="outer"))
                        s_flag_wide = (s_fl.to_frame() if s_flag_wide is None else s_flag_wide.join(s_fl, how="outer"))

            except Exception:
                pass  # skip failed symbols silently

        pbar.empty()
        status.empty()

        # Aggregate: flags → BreadthPct
        def _to_breadth(fw: Optional[pd.DataFrame]) -> pd.DataFrame:
            if fw is None:
                return pd.DataFrame(columns=["Date", "BreadthPct", "UpCount", "TotalCount"])
            fw = fw.sort_index()
            fw = fw[fw.count(axis=1) >= 1]           # drop dates with no data at all
            up = fw.fillna(0).sum(axis=1)
            pct_series = (up / n_symbols) * 100.0
            return (
                pd.DataFrame({
                    "Date":       pct_series.index,
                    "BreadthPct": pct_series.values.round(2),
                    "UpCount":    up.values.astype(int),
                    "TotalCount": np.full(len(up), n_symbols, dtype=int),
                })
                .dropna(subset=["BreadthPct"])
                .reset_index(drop=True)
            )

        breadth_df = _to_breadth(flag_wide)
        scanner_df = _to_breadth(s_flag_wide)

        # Persist to session state
        st.session_state.breadth_df = breadth_df
        st.session_state.scanner_df = scanner_df
        st.session_state.run_params = {
            "index_key":      index_key,
            "start_date":     str(start_date),
            "end_date":       str(end_date),
            "pct":            pct,
            "reversal_boxes": reversal_boxes,
            "xozone_window":  xozone_window,
            "n_symbols":      n_symbols,
        }

        st.success(
            f"✅ Done — {n_symbols} symbols → "
            f"{len(scanner_df)} trading days in scanner breadth"
        )

    # ─── Render chart from session state ──────────────────────
    # (runs on every re-run, including SMA slider moves)
    if "scanner_df" in st.session_state:
        breadth_df: pd.DataFrame = st.session_state.breadth_df
        scanner_df: pd.DataFrame = st.session_state.scanner_df
        params: dict             = st.session_state.run_params

        if scanner_df.empty:
            st.warning("No data returned. Check symbol universe and date range.")
            return

        chart_title = (
            f"{_short_label(params['index_key'])}  ·  "
            f"{params['start_date']} → {params['end_date']}"
        )
        fig = build_chart(breadth_df, scanner_df, sma_period, chart_title)
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary metrics ────────────────────────────────────
        last    = scanner_df.iloc[-1]
        sma_now = (
            scanner_df["BreadthPct"]
            .rolling(sma_period, min_periods=1)
            .mean()
            .iloc[-1]
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last Date",         str(last["Date"])[:10])
        c2.metric("Scanner Breadth",   f"{last['BreadthPct']:.1f}%")
        c3.metric(f"{sma_period}-SMA", f"{sma_now:.1f}%",
                  delta=f"{last['BreadthPct'] - sma_now:+.1f}%")
        c4.metric("Up / Total",        f"{int(last['UpCount'])} / {int(last['TotalCount'])}")
        if not breadth_df.empty:
            std_last = breadth_df.iloc[-1]["BreadthPct"]
            c5.metric("Std Breadth (L2)", f"{std_last:.1f}%")



    else:
        st.info(
            "👈 Configure settings in the sidebar and click **Run Analysis** to start.\n\n"
            "**Tip:** Start with *Nifty 50* (~30 sec) before trying larger indices."
        )


if __name__ == "__main__":
    main()
