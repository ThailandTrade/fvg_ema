#!/usr/bin/env python3
# mt5_bulk_fetch_to_pg.py
"""
MT5 -> Postgres ingestor (Dynamic UTC+2/UTC+3 offset) + EMA-50/EMA-200 + AlphaTrend SuperTrend (KivancOzbilgic)
- Inserts all *closed* candles <= cap.
- Fully reproduces PineScript logic:
    ATR = SMA(TR, AP)
    AlphaTrend := conditional upT/downT logic exactly as Pine
- Stores columns:
    ema_50, ema_200,
    st_upT, st_downT, st_alpha
"""

import os, re, csv, sys, time
import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Tuple

import MetaTrader5 as mt5
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, Float, select, desc, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.types import Numeric

UTC = timezone.utc

BATCH_BARS = 10000

# ---------- EMA CONSTANTS ----------
EMA_LEN_50 = 50
EMA_ALPHA_50 = Decimal("2") / Decimal(str(EMA_LEN_50 + 1))

EMA_LEN_200 = 200
EMA_ALPHA_200 = Decimal("2") / Decimal(str(EMA_LEN_200 + 1))

# --------- SUPERTREND CONSTANTS ----------
ST_AP = 14                       # Common Period (ATR length, RSI length)
ST_COEFF = Decimal("0.9")          # coeff = multiplier = 1.0 in Pine
ST_POS_THRESHOLD = Decimal("50") # 50 for RSI check

TF_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_100, "1d": 86_400_000, "1w": 604_800_000
}
TF_MT5 = {
    "1m": mt5.TIMEFRAME_M1, "3m": mt5.TIMEFRAME_M3, "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1,
    "2h": mt5.TIMEFRAME_H2, "4h": mt5.TIMEFRAME_H4, "6h": mt5.TIMEFRAME_H6,
    "8h": mt5.TIMEFRAME_H8, "12h": mt5.TIMEFRAME_H12,
    "1d": mt5.TIMEFRAME_D1, "1w": mt5.TIMEFRAME_W1
}

# ---------- DST helpers unchanged ----------
def get_last_sunday(year: int, month: int) -> datetime:
    d = datetime(year, month, 1, tzinfo=UTC) + timedelta(days=31)
    if d.month != month:
        d -= timedelta(days=d.day)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)

def get_server_offset_hours(utc_ms: int) -> int:
    dt_utc = datetime.fromtimestamp(utc_ms / 1000, tz=UTC)
    year = dt_utc.year
    dst_start = get_last_sunday(year, 3) + timedelta(hours=1)
    dst_end   = get_last_sunday(year, 10) + timedelta(hours=1)
    return 3 if dst_start <= dt_utc < dst_end else 2

def utc_ms_to_server_ms(utc_ms: int) -> int:
    return utc_ms + get_server_offset_hours(utc_ms) * 3600 * 1000

def server_ms_to_utc_ms(server_ms: int) -> int:
    approx = server_ms - 2 * 3600 * 1000
    return server_ms - get_server_offset_hours(approx) * 3600 * 1000

# ---------- UTILS ----------
def price_scale(base: str, quote: str) -> int:
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def iso_utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="seconds")

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = os.getenv("PG_PORT", "5432")
    db   = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd  = os.getenv("PG_PASSWORD", "postgres")
    return create_engine(
        f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=disable",
        pool_pre_ping=True,
        future=True
    )

def ensure_column(engine, table_name: str, col: str, scale: int):
    sql = text(f'ALTER TABLE IF EXISTS "{table_name}" ADD COLUMN IF NOT EXISTS {col} NUMERIC(20,{scale});')
    with engine.begin() as c:
        c.execute(sql)

def fetch_recent(engine, table, n: int, before_ts: Optional[int]):
    with engine.connect() as c:
        if before_ts is None:
            q = select(table.c.close).order_by(table.c.ts.asc()).limit(n)
        else:
            q = select(table.c.close).where(table.c.ts < before_ts).order_by(table.c.ts.desc()).limit(n)
        rows = c.execute(q).fetchall()
    vals = [Decimal(r.close) for r in rows]
    if before_ts:
        vals.reverse()
    return vals

def parse_pairs(path: str):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p:
                out.append(p.strip())
    return out

def parse_timeframes(path: str):
    tfs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip().lower()
            if t and not t.startswith("#"):
                tfs.append(t)
    return tfs

def mt5_now_server_ms_frozen():
    return utc_ms_to_server_ms(int(time.time() * 1000))

# ---------- SUPERTREND STATE ----------
class STState:
    def __init__(self):
        self.rsi_buffer = []
        self.tr_buffer  = []
        self.at_prev    = None  # AlphaTrend[n-1]

# ---------- MAIN INSERT FUNCTION WITH SUPERTREND ----------
def fetch_and_store(engine, pair, tf, user_to_ms, now_server_ms_fixed):
    base, quote = pair[:3], pair[3:]
    sym = None
    for cand in [pair, pair+".a", pair+".i", pair+".pro", pair+".ecn"]:
        if mt5.symbol_info(cand):
            mt5.symbol_select(cand, True)
            sym = cand
            break
    if not sym:
        print(f"[WARN] {pair}: not visible.")
        return

    meta = MetaData()
    scale = price_scale(base, quote)
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    table = Table(
        table_name, meta,
        Column("ts", BigInteger, primary_key=True),
        Column("ts_utc", String),
        Column("open", Numeric(20, scale)),
        Column("high", Numeric(20, scale)),
        Column("low",  Numeric(20, scale)),
        Column("close", Numeric(20, scale)),
        Column("volume", Float),
        Column("exchange", String(16)),
        Column("symbol", String(32)),
        Column("base", String(8)),
        Column("quote", String(8)),
        Column("timeframe", String(8)),
        Column("ema_50", Numeric(20, scale)),
        Column("ema_200", Numeric(20, scale)),
        Column("st_upT", Numeric(20, scale)),
        Column("st_downT", Numeric(20, scale)),
        Column("st_alpha", Numeric(20, scale))
    )
    meta.create_all(engine, checkfirst=True)

    ensure_column(engine, table_name, "st_upT", scale)
    ensure_column(engine, table_name, "st_downT", scale)
    ensure_column(engine, table_name, "st_alpha", scale)

    # -------- ST STATE --------
    st = STState()
    # ---- Retrieve last row (EMA + ST previous alpha) ----
    last_ts = last_close = last_ema50 = last_ema200 = None
    with engine.connect() as c:
        row = c.execute(
            select(table.c.ts, table.c.close, table.c.ema_50, table.c.ema_200, table.c.st_alpha)
            .order_by(desc(table.c.ts)).limit(1)
        ).fetchone()

        if row:
            last_ts     = int(row.ts)
            last_close  = Decimal(row.close)
            last_ema50  = Decimal(row.ema_50) if row.ema_50 else None
            last_ema200 = Decimal(row.ema_200) if row.ema_200 else None
            st.at_prev  = Decimal(row.st_alpha) if row.st_alpha is not None else None

    # ---------- Compute start window ----------
    if last_ts:
        start_utc = datetime.fromtimestamp((last_ts + 1) / 1000, tz=UTC)
        resume = True
    else:
        start_utc = datetime.now(UTC) - timedelta(days=365)
        resume = False

    start_ms = int(start_utc.timestamp() * 1000)
    offset   = get_server_offset_hours(start_ms)
    start_srv = (start_utc + timedelta(hours=offset)).replace(tzinfo=None)

    tf_ms = TF_MS[tf]
    cap_now = (now_server_ms_fixed // tf_ms) * tf_ms - tf_ms

    if user_to_ms:
        cap_to = (utc_ms_to_server_ms(user_to_ms) // tf_ms) * tf_ms - tf_ms
        cap = min(cap_now, cap_to)
    else:
        cap = cap_now

    end_srv = datetime.fromtimestamp(cap / 1000).replace(tzinfo=None) + timedelta(seconds=1)

    if start_srv >= end_srv:
        print(f"[INFO] {pair} {tf}: no new bars.")
        return

    # -------- EMA seed buffers --------
    ema_prev50 = last_ema50
    ema_prev200 = last_ema200
    seed50 = []
    seed200 = []

    if ema_prev50 is None:
        seed50 = fetch_recent(engine, table, EMA_LEN_50 - 1, before_ts=last_ts + 1 if last_ts else None)

    if ema_prev200 is None:
        seed200 = fetch_recent(engine, table, EMA_LEN_200 - 1, before_ts=last_ts + 1 if last_ts else None)

    inserted = 0
    cur = start_srv
    prev_close = last_close

    with engine.begin() as conn:
        while cur < end_srv:
            end = cur + timedelta(milliseconds=tf_ms * BATCH_BARS)
            if end > end_srv:
                end = end_srv

            rates = mt5.copy_rates_range(sym, TF_MT5[tf], cur, end)
            if rates is None or len(rates) == 0:
                cur = end
                continue

            rows = []
            for r in rates:
                bar_srv = int(r["time"]) * 1000
                if bar_srv > cap:
                    continue

                ts_utc = server_ms_to_utc_ms(bar_srv)
                if user_to_ms and ts_utc > user_to_ms:
                    continue
                if last_ts and ts_utc <= last_ts:
                    continue

                # ------- OHLC -------
                mt5_o = qround(r["open"], scale)
                mt5_h = qround(r["high"], scale)
                mt5_l = qround(r["low"], scale)
                mt5_c = qround(r["close"], scale)

                o = prev_close if prev_close is not None else mt5_o
                c = mt5_c
                h = max(o, c, mt5_h)
                l = min(o, c, mt5_l)
                v = float(r["tick_volume"])

                # ------- EMA50 -------
                ema50 = None
                if ema_prev50 is not None:
                    ema50 = qround(EMA_ALPHA_50 * c + (Decimal(1) - EMA_ALPHA_50) * ema_prev50, scale)
                    ema_prev50 = ema50
                else:
                    seed50.append(c)
                    if len(seed50) == EMA_LEN_50:
                        sma = qround(sum(seed50) / Decimal(EMA_LEN_50), scale)
                        ema_prev50 = sma
                        ema50 = sma

                # ------- EMA200 -------
                ema200 = None
                if ema_prev200 is not None:
                    ema200 = qround(EMA_ALPHA_200 * c + (Decimal(1) - EMA_ALPHA_200) * ema_prev200, scale)
                    ema_prev200 = ema200
                else:
                    seed200.append(c)
                    if len(seed200) == EMA_LEN_200:
                        sma = qround(sum(seed200) / Decimal(EMA_LEN_200), scale)
                        ema_prev200 = sma
                        ema200 = sma

                # =====================================================
                #   SUPERTREND EXACT PINE (AlphaTrend)
                # =====================================================

                # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
                if prev_close is None:
                    tr = mt5_h - mt5_l
                else:
                    tr = max(
                        h - l,
                        abs(h - prev_close),
                        abs(l - prev_close)
                    )
                st.tr_buffer.append(tr)

                # SMA(TR, AP)
                if len(st.tr_buffer) > ST_AP:
                    st.tr_buffer.pop(0)

                if len(st.tr_buffer) == ST_AP:
                    ATR = sum(st.tr_buffer) / Decimal(ST_AP)
                else:
                    ATR = None

                # RSI(src, AP) version EXACT Pine (simple buffer-based RSI)
                st.rsi_buffer.append(c)
                if len(st.rsi_buffer) > ST_AP + 1:
                    st.rsi_buffer.pop(0)

                RSI = None
                if len(st.rsi_buffer) == ST_AP + 1:
                    gains = []
                    losses = []
                    for i in range(1, len(st.rsi_buffer)):
                        diff = st.rsi_buffer[i] - st.rsi_buffer[i-1]
                        if diff > 0:
                            gains.append(diff)
                        else:
                            losses.append(-diff)
                    avg_gain = sum(gains) / Decimal(ST_AP)
                    avg_loss = sum(losses) / Decimal(ST_AP)
                    if avg_loss == 0:
                        RSI = Decimal(100)
                    else:
                        RS = avg_gain / avg_loss
                        RSI = Decimal(100) - (Decimal(100) / (Decimal(1) + RS))

                upT = downT = None
                alpha = None

                if ATR is not None:
                    upT = l - ATR * ST_COEFF
                    downT = h + ATR * ST_COEFF

                    cond = (RSI is not None and RSI >= ST_POS_THRESHOLD)

                    if cond:
                        if upT < (st.at_prev if st.at_prev is not None else upT):
                            alpha = st.at_prev
                        else:
                            alpha = upT
                    else:
                        if downT > (st.at_prev if st.at_prev is not None else downT):
                            alpha = st.at_prev
                        else:
                            alpha = downT

                st.at_prev = alpha

                rows.append({
                    "ts": ts_utc,
                    "ts_utc": iso_utc(ts_utc),
                    "open": o, "high": h, "low": l, "close": c,
                    "volume": v,
                    "exchange": "mt5",
                    "symbol": sym,
                    "base": base,
                    "quote": quote,
                    "timeframe": tf,
                    "ema_50": ema50,
                    "ema_200": ema200,
                    "st_upT": upT,
                    "st_downT": downT,
                    "st_alpha": alpha
                })

                prev_close = c
                last_ts = ts_utc

            if rows:
                res = conn.execute(
                    pg_insert(table).values(rows).on_conflict_do_nothing(index_elements=["ts"])
                )
                inserted += (res.rowcount or 0)

            cur = end

    print(f"[DONE] {pair} {tf}: +{inserted} candles inserted with EMA + SuperTrend.")


# ============================
# MAIN LOOP
# ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to", default=None)
    ap.add_argument("--pairs-file", default=os.getenv("PAIRS_FILE", "pairs.txt"))
    ap.add_argument("--timeframes-file", default=os.getenv("TIMEFRAMES_FILE", "timeframes.txt"))
    ap.add_argument("--pairs", type=str, default=None)
    args = ap.parse_args()

    user_to_ms = None
    if args.to:
        s = args.to.strip()
        if re.fullmatch(r"\d{10,13}", s):
            if len(s) == 10:
                s += "000"
            user_to_ms = int(s)
        else:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            user_to_ms = int(datetime.fromisoformat(s).timestamp() * 1000)

    if not mt5.initialize():
        print("[ERR] MT5 init failed")
        sys.exit(1)

    engine = get_pg_engine()

    if args.pairs:
        pairs = [x for x in re.split(r"[,\s]+", args.pairs.strip()) if len(x) >= 6]
    else:
        pairs = parse_pairs(args.pairs_file)

    tfs = parse_timeframes(args.timeframes_file)

    print("[INIT] Live ingestion — EMA + SuperTrend")

    try:
        while True:
            now_srv = mt5_now_server_ms_frozen()
            offset = get_server_offset_hours(int(time.time() * 1000))
            print(f"[LOOP] server≈ {iso_utc(now_srv)} (offset=+{offset}h)")

            for p in pairs:
                for tf in tfs:
                    fetch_and_store(engine, p, tf, user_to_ms, now_srv)

            time.sleep(1)

    except KeyboardInterrupt:
        print("[STOP] User interrupt.")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
