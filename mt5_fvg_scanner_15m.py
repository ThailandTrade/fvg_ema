#!/usr/bin/env python3
# mt5_fvg_scanner_15m_v3.py
"""
MT5 -> FVG/EMA50 Scanner (15M Only) - ADJUSTABLE FILTER
- Fetches the LATEST CLOSED CANDLE (15m).
- Checks for FVG.
- FILTERS FVG based on Standard Deviation.
- Checks trend via EMA-50.
- Prints "Detected Setup" ONCE per candle per pair.
"""

import os, re, csv, sys, time
import argparse
import statistics
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List

import MetaTrader5 as mt5

UTC = timezone.utc

# ---------- CONFIG ----------
EMA_LEN_50 = 50
EMA_ALPHA_50 = Decimal("2") / Decimal(str(EMA_LEN_50 + 1))
STDEV_PERIOD = 200 # Période pour calculer la volatilité moyenne des gaps

SCAN_TF_MT5 = mt5.TIMEFRAME_M15

# ---------- UTILS ----------
def price_scale(base: str, quote: str) -> int:
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime('%Y-%m-%d %H:%M')

def parse_pairs(path: str):
    out = []
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p:
                out.append(p.strip())
    return out

# ---------- LOGIC ----------
def calculate_ema_series(rates, length=50) -> Optional[Decimal]:
    if len(rates) < length:
        return None
    
    closes = [Decimal(str(r["close"])) for r in rates]
    sma = sum(closes[:length]) / Decimal(length)
    ema = sma
    alpha = Decimal("2") / Decimal(str(length + 1))

    for c in closes[length:]:
        ema = alpha * c + (Decimal("1") - alpha) * ema
    
    return ema

def check_filtered_fvg(rates, threshold: float) -> Tuple[bool, bool, float]:
    """
    Checks for FVG and applies the StDev filter.
    Returns: (is_bullish, is_bearish, score)
    """
    if len(rates) < STDEV_PERIOD + 3:
        return False, False, 0.0

    # Indices:
    # -1 = Current closed candle (n)
    # -2 = Middle candle (n-1)
    # -3 = Left candle (n-2)

    c1 = rates[-3]["close"]
    h1 = rates[-3]["high"]
    l1 = rates[-3]["low"]
    
    c3 = rates[-1]["close"]
    h3 = rates[-1]["high"]
    l3 = rates[-1]["low"]

    # Basic Geometry
    raw_bull_cond = (h1 < l3) and (c3 > c1)
    raw_bear_cond = (l1 > h3) and (c3 < c1)

    if not raw_bull_cond and not raw_bear_cond:
        return False, False, 0.0

    # --- StDev Calculation (Optimized for speed) ---
    # We need the series of (Low[i] - High[i-2]) for bullish ref
    # and (Low[i-2] - High[i]) for bearish ref
    
    subset = rates[-(STDEV_PERIOD + 5):]
    
    # Pre-calculate lists to avoid loop overhead inside stats
    lows = [r["low"] for r in subset]
    highs = [r["high"] for r in subset]
    
    # We need about 200 diffs.
    # Logic: gap_size = lows[i] - highs[i-2]
    
    diffs = []
    # Calculate generalized gap volatility (absolute values of gaps to gauge market "gappiness")
    # This is a simplification of the Pine script to make it robust:
    # We just want to know: "What is a standard gap size recently?"
    
    for i in range(2, len(lows)):
        gap = abs(lows[i] - highs[i-2])
        diffs.append(gap)

    recent_diffs = diffs[-STDEV_PERIOD:]
    if len(recent_diffs) < 2: 
        return False, False, 0.0
        
    volatility = statistics.stdev(recent_diffs)
    if volatility == 0: volatility = 1.0e-5 # Avoid div/0

    # --- Apply Filter ---
    is_bullish = False
    is_bearish = False
    score = 0.0

    if raw_bull_cond:
        current_gap = l3 - h1
        score = current_gap / volatility
        if score > threshold:
            is_bullish = True

    elif raw_bear_cond:
        current_gap = l1 - h3
        score = current_gap / volatility
        if score > threshold:
            is_bearish = True

    return is_bullish, is_bearish, score

def scan_pair(pair: str, last_alerts: Dict[str, int], threshold: float):
    if not mt5.symbol_select(pair, True):
        return

    # Fetch enough history
    rates = mt5.copy_rates_from_pos(pair, SCAN_TF_MT5, 1, 450)

    if rates is None or len(rates) < 250:
        return

    last_candle_ts = int(rates[-1]["time"])

    # Skip duplicate alerts
    if last_alerts.get(pair) == last_candle_ts:
        return

    base, quote = pair[:3], pair[3:]
    scale = price_scale(base, quote)

    # 1. EMA 50
    ema_50 = calculate_ema_series(rates, EMA_LEN_50)
    if ema_50 is None:
        return

    current_close = qround(rates[-1]["close"], scale)
    
    # 2. Filtered FVG Check
    is_bull_fvg, is_bear_fvg, score = check_filtered_fvg(rates, threshold)

    setup_msg = None

    # 3. Combine Logic
    if is_bull_fvg and current_close > ema_50:
        setup_msg = f"LONG (Score: {score:.2f})"
    elif is_bear_fvg and current_close < ema_50:
        setup_msg = f"SHORT (Score: {score:.2f})"

    # 4. Print
    if setup_msg:
        ts_str = format_ts(last_candle_ts)
        print(f"Detected Setup : {pair} {setup_msg} | Bougie : {ts_str}")
        last_alerts[pair] = last_candle_ts

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-file", default="pairs.txt")
    # Ajout de l'argument seuil. Défaut 0.2 (plus permissif que 0.5)
    ap.add_argument("--threshold", type=float, default=0.2, help="Seuil de sensibilité FVG (Ex: 0.1=Très sensible, 0.5=Strict, 1.0=Massif uniquement)")
    args = ap.parse_args()

    if not mt5.initialize():
        print("MT5 Init Failed")
        sys.exit(1)

    pairs = parse_pairs(args.pairs_file)
    if not pairs:
        print("No pairs found in pairs.txt")
        mt5.shutdown()
        sys.exit(1)

    print(f"Scanning {len(pairs)} pairs on 15M...")
    print(f"Filter Threshold: {args.threshold} (GapSize > {args.threshold} * StDev)")
    
    last_alerts = {}

    try:
        while True:
            for p in pairs:
                scan_pair(p, last_alerts, args.threshold)
            time.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()