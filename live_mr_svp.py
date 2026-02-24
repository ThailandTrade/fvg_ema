#!/usr/bin/env python3
"""
Live Trading - Combined MR + CB Strategy
ALIGNÉ EXACTEMENT SUR LE BACKTEST combined_mr_breakout
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import psycopg2
import os
import json
import time
import logging
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from collections import defaultdict
from decimal import Decimal, ROUND_DOWN
from sqlalchemy import create_engine

load_dotenv()

# =============================================================================
# CONFIGURATION - ALIGNÉ SUR BACKTEST combined_mr_breakout
# =============================================================================

# Strategy enable/disable
ENABLE_MR = True          # Mean Reversion after reintegration
ENABLE_CB = True          # Confirmed Breakout after sustained breakout

RISK_PERCENT = 0.001  # 0.1% par trade
WAIT_CANDLES = 3      # Candles to wait after breakout before deciding MR or CB

# =============================================================================
# MR-SPECIFIC CONFIG (Mean Reversion)
# =============================================================================
MR_MIN_RR = 1.5
MR_SL_OFFSET = 1.0
MR_TP1_RR = 1.5
MR_TP1_SPLIT = 0.7
MR_TP2_SPLIT = 0.3
MR_USE_TRAILING = True
MR_MIN_POC_STRENGTH = 2.0
MR_FILTER_ENTRY_VS_POC = True
MR_MAX_BREAKOUT_DURATION_MINUTES = 3
MR_EXCLUDED_HOURS = []

# =============================================================================
# CB-SPECIFIC CONFIG (Confirmed Breakout)
# =============================================================================
CB_MIN_RR = 2.0
CB_SL_OFFSET = 0.75
CB_TP1_RR = 1.0
CB_TP1_SPLIT = 0.3
CB_TP2_SPLIT = 0.7
CB_USE_TRAILING = True
CB_MIN_POC_STRENGTH = 3.0
CB_EXCLUDED_HOURS = [0, 10]
CB_EXCLUDE_VAH_TARGET = True
CB_USE_PREV_DAY = True
CB_USE_PREV_WEEK = True

# Shared filters
USE_VP_SHAPE_FILTER = True
EXCLUDED_VP_SHAPES = [""]

# Reset VP
RESET_VP_PER_SESSION = True

# =============================================================================
# CONFIGURATION DES HEURES DE SESSION (UTC)
# vp_start/vp_end : heures pour collecter les ticks et construire le VP
# trade_start/trade_end : heures où les entrées en position sont autorisées
# =============================================================================
SESSIONS_CONFIG = {
    'TOKYO':  {'vp_start': 0,    'vp_end': 4,    'trade_start': 0,    'trade_end': 4},
    'LONDON': {'vp_start': 8,    'vp_end': 14.5,   'trade_start': 9,    'trade_end': 14},
    'NY':     {'vp_start': 14.5, 'vp_end': 21.5,   'trade_start': 15, 'trade_end': 21},
}

# =============================================================================
# CONFIGURATION PAR ASSET
# =============================================================================
ASSETS = [
    {
        'enabled': True,
        'symbol': 'XAUUSD',
        'mt5_symbol': 'XAUUSD',
        'candle_table': 'candles_mt5_xauusd_1m',
        'tick_table': 'market_ticks_xauusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3, 4],
    },{
        'enabled': False,
        'symbol': 'XAGUSD',
        'mt5_symbol': 'XAGUSD',
        'candle_table': 'candles_mt5_xagusd_1m',
        'tick_table': 'market_ticks_xagusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'JP225.cash',
        'mt5_symbol': 'JP225.cash',
        'candle_table': 'candles_mt5_jp225_cash_1m',
        'tick_table': 'market_ticks_jp225',
        'tick_size': 1.0,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': True},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'UK100.cash',
        'mt5_symbol': 'UK100.cash',
        'candle_table': 'candles_mt5_uk100_cash_1m',
        'tick_table': 'market_ticks_uk100',
        'tick_size': 1.0,
        'va_percent': 0.70,
        'allow_long': False,
        'allow_short': True,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'GER40.cash',
        'mt5_symbol': 'GER40.cash',
        'candle_table': 'candles_mt5_ger40_cash_1m',
        'tick_table': 'market_ticks_ger40',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': False, 'NY': True},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'US30.cash',
        'mt5_symbol': 'US30.cash',
        'candle_table': 'candles_mt5_us30_cash_1m',
        'tick_table': 'market_ticks_us30',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': False,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'XAUAUD',
        'mt5_symbol': 'XAUAUD',
        'candle_table': 'candles_mt5_xauaud_1m',
        'tick_table': 'market_ticks_xauaud',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': False,
        'allow_short': True,
        'sessions': {'TOKYO': False, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'XPDUSD',
        'mt5_symbol': 'XPDUSD',
        'candle_table': 'candles_mt5_xpdusd_1m',
        'tick_table': 'market_ticks_xpdusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': True},
        'allowed_days': [0, 1, 2, 3, 4],
    },
    {
        'enabled': False,
        'symbol': 'XPTUSD',
        'mt5_symbol': 'XPTUSD',
        'candle_table': 'candles_mt5_xptusd_1m',
        'tick_table': 'market_ticks_xptusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': False, 'NY': True},
        'allowed_days': [0, 1, 2, 3, 4],
    },
]

# =============================================================================
# PARAMETRES LIVE
# =============================================================================
LOOP_INTERVAL_SECONDS = 5
STATE_FILE = "combined_mr_cb_state.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATABASE
# =============================================================================

def get_pg_engine():
    db_url = f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DB')}"
    return create_engine(db_url)


def fetch_candles_from_db(engine, table_name: str, limit: int = 60):
    query = f"SELECT ts, open, high, low, close FROM {table_name} ORDER BY ts DESC LIMIT {limit}"
    df = pd.read_sql(query, engine)
    if df.empty:
        return []
    df = df.sort_values('ts').reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df.to_dict('records')


def fetch_ticks_from_db(engine, tick_table: str, start_time: datetime, end_time: datetime):
    """
    Récupère les ticks de start_time (inclus) à end_time (EXCLUS)
    Aligné sur le backtest qui utilise floor('T') pour grouper par minute
    """
    query = f"""
        SELECT time, last as price, volume 
        FROM {tick_table} 
        WHERE time >= '{start_time.strftime("%Y-%m-%d %H:%M:%S")}' 
        AND time < '{end_time.strftime("%Y-%m-%d %H:%M:%S")}'
        ORDER BY time ASC
    """
    df = pd.read_sql(query, engine)
    if df.empty:
        return np.array([]), np.array([])
    return df['price'].values, df['volume'].values


# =============================================================================
# VOLUME PROFILE
# =============================================================================

class IncrementalVolumeProfile:
    def __init__(self, tick_size: float = 0.01, va_percent: float = 0.70):
        self.tick_size = tick_size
        self.va_percent = va_percent
        self.reset()
    
    def reset(self):
        self.profile = defaultdict(float)
        self.total_volume = 0.0
        self._cache_valid = False
        self._cached_poc = None
        self._cached_vah = None
        self._cached_val = None
        self._cached_poc_strength = None
    
    def add_ticks(self, prices: np.ndarray, volumes: np.ndarray):
        if len(prices) == 0:
            return
        price_bins = np.round(prices / self.tick_size) * self.tick_size
        unique_bins, indices = np.unique(price_bins, return_inverse=True)
        bin_volumes = np.bincount(indices, weights=volumes)
        for price_bin, vol in zip(unique_bins, bin_volumes):
            self.profile[price_bin] += vol
            self.total_volume += vol
        self._cache_valid = False
    
    def _compute_value_area(self):
        if not self.profile:
            self._cached_poc = self._cached_vah = self._cached_val = None
            self._cached_poc_strength = None
            self._cache_valid = True
            return
        sorted_bins = sorted(self.profile.keys())
        volumes = np.array([self.profile[b] for b in sorted_bins])
        poc_idx = np.argmax(volumes)
        self._cached_poc = sorted_bins[poc_idx]
        poc_volume = volumes[poc_idx]
        avg_volume = np.mean(volumes)
        self._cached_poc_strength = poc_volume / avg_volume if avg_volume > 0 else 0
        target_volume = self.total_volume * self.va_percent
        current_volume = volumes[poc_idx]
        up_idx, down_idx = poc_idx + 1, poc_idx - 1
        while current_volume < target_volume:
            vol_up = volumes[up_idx] if up_idx < len(volumes) else 0
            vol_down = volumes[down_idx] if down_idx >= 0 else 0
            if vol_up == 0 and vol_down == 0:
                break
            if vol_up >= vol_down:
                current_volume += vol_up
                up_idx += 1
            else:
                current_volume += vol_down
                down_idx -= 1
        self._cached_vah = sorted_bins[min(up_idx - 1, len(sorted_bins) - 1)]
        self._cached_val = sorted_bins[max(down_idx + 1, 0)]
        self._cache_valid = True
    
    def get_levels(self):
        if not self._cache_valid:
            self._compute_value_area()
        return self._cached_poc, self._cached_vah, self._cached_val
    
    def get_poc_strength(self):
        if not self._cache_valid:
            self._compute_value_area()
        return self._cached_poc_strength
    
    def get_profile_shape(self):
        if not self._cache_valid:
            self._compute_value_area()
        if self._cached_poc_strength is None or self._cached_poc_strength < 1.3:
            return "FLAT"
        if self._cached_poc is None or self._cached_vah is None or self._cached_val is None:
            return "FLAT"
        va_range = self._cached_vah - self._cached_val
        if va_range == 0:
            return "FLAT"
        poc_position = (self._cached_poc - self._cached_val) / va_range
        if poc_position > 0.66:
            return "P-SHAPE"
        elif poc_position < 0.33:
            return "B-SHAPE"
        else:
            return "D-SHAPE"

    def snapshot(self):
        poc, vah, val = self.get_levels()
        return {'poc': poc, 'vah': vah, 'val': val}


# =============================================================================
# STRUCTURAL LEVELS TRACKER (for CB targets)
# =============================================================================

class StructuralLevelsTracker:
    def __init__(self, tick_size=0.01, va_percent=0.70):
        self.tick_size = tick_size
        self.va_percent = va_percent
        self._daily_vp = IncrementalVolumeProfile(tick_size, va_percent)
        self._weekly_vp = IncrementalVolumeProfile(tick_size, va_percent)
        self.prev_day = {'poc': None, 'vah': None, 'val': None}
        self.prev_week = {'poc': None, 'vah': None, 'val': None}
        self._current_day = None
        self._current_week_start = None

    def update(self, dt, prices, volumes):
        day = dt.date()
        week_start = (dt - timedelta(days=dt.weekday())).date()
        if self._current_day is not None and day != self._current_day:
            snap = self._daily_vp.snapshot()
            if snap['poc'] is not None:
                self.prev_day = snap
            self._daily_vp.reset()
        if self._current_week_start is not None and week_start != self._current_week_start:
            snap = self._weekly_vp.snapshot()
            if snap['poc'] is not None:
                self.prev_week = snap
            self._weekly_vp.reset()
        self._current_day = day
        self._current_week_start = week_start
        if len(prices) > 0:
            self._daily_vp.add_ticks(prices, volumes)
            self._weekly_vp.add_ticks(prices, volumes)

    def get_target_levels(self, direction: str, entry_price: float):
        candidates = []
        if CB_USE_PREV_DAY:
            if self.prev_day['vah'] is not None and not CB_EXCLUDE_VAH_TARGET:
                candidates.append(('PD_VAH', self.prev_day['vah']))
            if self.prev_day['val'] is not None:
                candidates.append(('PD_VAL', self.prev_day['val']))
            if self.prev_day['poc'] is not None:
                candidates.append(('PD_POC', self.prev_day['poc']))
        if CB_USE_PREV_WEEK:
            if self.prev_week['vah'] is not None and not CB_EXCLUDE_VAH_TARGET:
                candidates.append(('PW_VAH', self.prev_week['vah']))
            if self.prev_week['val'] is not None:
                candidates.append(('PW_VAL', self.prev_week['val']))
            if self.prev_week['poc'] is not None:
                candidates.append(('PW_POC', self.prev_week['poc']))
        if direction == 'LONG':
            valid = [(label, price) for label, price in candidates if price > entry_price]
            valid.sort(key=lambda x: x[1])
        else:
            valid = [(label, price) for label, price in candidates if price < entry_price]
            valid.sort(key=lambda x: -x[1])
        return valid


# =============================================================================
# SESSION HELPERS
# =============================================================================

def get_session(dt):
    """Retourne le nom de la session VP active pour un datetime donné."""
    h = dt.hour + dt.minute / 60.0
    for sess_name, cfg in SESSIONS_CONFIG.items():
        if cfg['vp_start'] <= h < cfg['vp_end']:
            return sess_name
    return "AUTRE"


def is_session_start(dt):
    """Vérifie si le datetime correspond au début exact d'une session VP."""
    current_time = dt.hour + dt.minute / 60.0
    for sess_name, cfg in SESSIONS_CONFIG.items():
        if current_time == cfg['vp_start']:
            return sess_name
    return None


def can_trade_now(dt, session_name: str) -> bool:
    """Vérifie si on peut trader à ce moment (dans les heures trade de la session)."""
    if session_name not in SESSIONS_CONFIG:
        return False
    cfg = SESSIONS_CONFIG[session_name]
    current_time = dt.hour + dt.minute / 60.0
    return cfg['trade_start'] <= current_time < cfg['trade_end']


# =============================================================================
# MT5 HELPERS
# =============================================================================

def qround(price: float, tick_size: float) -> float:
    d = Decimal(str(price))
    t = Decimal(str(tick_size))
    return float(d.quantize(t, rounding=ROUND_DOWN))


def get_lot_size(symbol: str, entry: float, sl: float, risk_amount: float) -> float:
    if entry == sl:
        return 0.0
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logger.error(f"[{symbol}] Symbol info not found")
        return 0.0
    
    if entry > sl:
        order_type = mt5.ORDER_TYPE_BUY
    else:
        order_type = mt5.ORDER_TYPE_SELL
    
    try:
        profit_one_lot = mt5.order_calc_profit(order_type, symbol, 1.0, float(entry), float(sl))
    except Exception as e:
        logger.error(f"[{symbol}] order_calc_profit error: {e}")
        return 0.0
    
    if profit_one_lot is None:
        logger.warning(f"[{symbol}] Fallback manual lot calculation")
        loss_per_lot = abs(entry - sl) * symbol_info.trade_contract_size
    else:
        loss_per_lot = abs(profit_one_lot)
    
    if loss_per_lot == 0:
        return 0.0
    
    raw_lots = risk_amount / loss_per_lot
    
    step = symbol_info.volume_step
    min_vol = symbol_info.volume_min
    max_vol = symbol_info.volume_max
    
    lots = round(raw_lots / step) * step
    
    if lots < min_vol:
        lots = min_vol
    if lots > max_vol:
        lots = max_vol
    
    return float(lots)


def has_position_or_order(symbol: str) -> bool:
    positions = mt5.positions_get(symbol=symbol)
    if positions and len(positions) > 0:
        return True
    orders = mt5.orders_get(symbol=symbol)
    if orders and len(orders) > 0:
        return True
    return False


def manage_tp1_to_be(symbol: str, tick_size: float):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    our_positions = [p for p in positions if any(tag in (p.comment or "") for tag in ("VP_MR", "VP_CB"))]
    tp2_positions = [p for p in our_positions if "_TP2" in (p.comment or "")]
    tp1_positions = [p for p in our_positions if "_TP1" in (p.comment or "")]

    if len(tp1_positions) == 0 and len(tp2_positions) == 1:
        pos = tp2_positions[0]
        entry_price = pos.price_open
        current_sl = pos.sl

        if abs(current_sl - entry_price) > tick_size * 2:
            new_sl = qround(entry_price, tick_size)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }

            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[{symbol}] TP1 hit - SL moved to BE @ {new_sl}")
            else:
                logger.error(f"[{symbol}] Failed to move SL to BE: {result.retcode}")


def place_market_order(symbol: str, order_type: str, entry: float, sl: float, tp1: float, tp2: float,
                       risk_amount: float, tick_size: float,
                       tp1_split: float = 0.5, tp2_split: float = 0.5,
                       strategy: str = "VP_MR") -> bool:
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"[{symbol}] Symbol info not found")
        return False

    sl = qround(sl, tick_size)
    tp1 = qround(tp1, tick_size)
    tp2 = qround(tp2, tick_size)

    lot1 = get_lot_size(symbol, entry, sl, risk_amount * tp1_split)
    lot2 = get_lot_size(symbol, entry, sl, risk_amount * tp2_split)

    if lot1 <= 0 or lot2 <= 0:
        logger.error(f"[{symbol}] Invalid lot size: lot1={lot1}, lot2={lot2}")
        return False

    mt5_type = mt5.ORDER_TYPE_BUY if order_type == "BUY" else mt5.ORDER_TYPE_SELL
    price = info.ask if order_type == "BUY" else info.bid

    logger.info(f"[{symbol}] ====== PLACING {strategy} TRADE ======")
    logger.info(f"[{symbol}] Direction: {order_type} | Entry: {price} | SL: {sl}")
    logger.info(f"[{symbol}] TP1: {tp1} | TP2: {tp2}")
    logger.info(f"[{symbol}] Risk: ${risk_amount:.2f} | Split: {tp1_split:.0%}/{tp2_split:.0%} | Lot1: {lot1} | Lot2: {lot2}")

    success_count = 0

    request1 = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot1,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp1,
        "deviation": 20,
        "magic": 123456,
        "comment": f"{strategy}_TP1",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result1 = mt5.order_send(request1)
    if result1.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{symbol}] TP1 FILLED: {order_type} {lot1} lots @ {result1.price}")
        success_count += 1
    else:
        logger.error(f"[{symbol}] TP1 FAILED: {result1.retcode} - {result1.comment}")

    request2 = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot2,
        "type": mt5_type,
        "price": price,
        "sl": sl,
        "tp": tp2,
        "deviation": 20,
        "magic": 123456,
        "comment": f"{strategy}_TP2",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result2 = mt5.order_send(request2)
    if result2.retcode == mt5.TRADE_RETCODE_DONE:
        logger.info(f"[{symbol}] TP2 FILLED: {order_type} {lot2} lots @ {result2.price}")
        success_count += 1
    else:
        logger.error(f"[{symbol}] TP2 FAILED: {result2.retcode} - {result2.comment}")

    logger.info(f"[{symbol}] ====== TRADE COMPLETE: {success_count}/2 orders filled ======")
    return success_count > 0


def daily_market_close_guard(candle_dt: datetime) -> bool:
    if candle_dt.weekday() == 4 and candle_dt.hour == 21 and candle_dt.minute >= 50:
        logger.warning("Friday market close - closing all")
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "deviation": 20,
                    "magic": 123456,
                    "comment": "Friday close",
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
        orders = mt5.orders_get()
        if orders:
            for order in orders:
                mt5.order_send({"action": mt5.TRADE_ACTION_REMOVE, "order": order.ticket})
        return True
    return False


# =============================================================================
# ASSET STATE
# =============================================================================

class AssetState:
    def __init__(self, config: dict):
        self.config = config
        self.symbol = config['symbol']
        self.mt5_symbol = config['mt5_symbol']
        self.vp = IncrementalVolumeProfile(tick_size=config['tick_size'], va_percent=config['va_percent'])
        self.structural = StructuralLevelsTracker(tick_size=config['tick_size'], va_percent=config['va_percent'])
        self.state = "INSIDE"
        self.breakout_direction = None
        self.breakout_time = None
        self.candles_since_breakout = 0
        self.wait_highs = []
        self.wait_lows = []
        self.ghost_trade = None
        self.current_session = None
        self.last_trade_candle_ts = 0
        self.last_candle_ts = 0

    def reset_vp(self):
        self.vp.reset()
        self.state = "INSIDE"
        self.breakout_direction = None
        self.breakout_time = None
        self.candles_since_breakout = 0
        self.wait_highs = []
        self.wait_lows = []
        self.ghost_trade = None

    def to_dict(self) -> dict:
        return {
            'state': self.state,
            'breakout_direction': self.breakout_direction,
            'breakout_time': self.breakout_time.isoformat() if self.breakout_time else None,
            'candles_since_breakout': self.candles_since_breakout,
            'wait_highs': self.wait_highs,
            'wait_lows': self.wait_lows,
            'ghost_trade': self.ghost_trade,
            'current_session': self.current_session,
            'last_trade_candle_ts': self.last_trade_candle_ts,
            'last_candle_ts': self.last_candle_ts,
        }

    def from_dict(self, data: dict):
        self.state = data.get('state', 'INSIDE')
        self.breakout_direction = data.get('breakout_direction')
        bt = data.get('breakout_time')
        self.breakout_time = datetime.fromisoformat(bt) if bt else None
        self.candles_since_breakout = data.get('candles_since_breakout', 0)
        self.wait_highs = data.get('wait_highs', [])
        self.wait_lows = data.get('wait_lows', [])
        self.ghost_trade = data.get('ghost_trade')
        self.current_session = data.get('current_session')
        self.last_trade_candle_ts = data.get('last_trade_candle_ts', 0)
        self.last_candle_ts = data.get('last_candle_ts', 0)


def save_all_states(asset_states: dict):
    data = {symbol: state.to_dict() for symbol, state in asset_states.items()}
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def load_all_states(asset_states: dict):
    if not os.path.exists(STATE_FILE):
        logger.info("No state file found, starting fresh")
        return
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        for symbol, saved in data.items():
            if symbol in asset_states:
                asset_states[symbol].from_dict(saved)
                logger.info(f"[{symbol}] State restored: {saved['state']} | swing: {saved['swing_extreme']}")
    except Exception as e:
        logger.error(f"Failed to load state: {e}")


# =============================================================================
# MAIN TRADING LOGIC - ALIGNÉ SUR BACKTEST v3
# =============================================================================

def detect_and_trade(asset_state: AssetState, engine, account_balance: float) -> dict:
    config = asset_state.config
    symbol = config['symbol']
    mt5_symbol = config['mt5_symbol']

    result = {
        'new_candle': False,
        'symbol': symbol,
        'candle_dt': None,
        'close': None,
        'high': None,
        'low': None,
        'session': None,
        'state_before': asset_state.state,
        'state_after': None,
        'poc': None,
        'vah': None,
        'val': None,
        'poc_strength': None,
        'vp_shape': None,
        'event': None,
        'event_details': None,
        'has_position': False,
    }

    # Check position existante
    if has_position_or_order(mt5_symbol):
        result['has_position'] = True
        result['event'] = "POSITION_ACTIVE"
        return result

    # Charger les candles
    candles = fetch_candles_from_db(engine, config['candle_table'], limit=5)
    if not candles:
        result['event'] = "NO_DATA"
        return result

    last_candle = candles[-1]
    close = last_candle['close']
    high = last_candle['high']
    low = last_candle['low']
    candle_dt = last_candle['dt']
    candle_ts = last_candle['ts']

    result['candle_dt'] = candle_dt
    result['close'] = close
    result['high'] = high
    result['low'] = low

    # Vérifier si c'est une nouvelle bougie
    if asset_state.last_candle_ts == candle_ts:
        result['new_candle'] = False
        return result

    result['new_candle'] = True
    asset_state.last_candle_ts = candle_ts

    # ======================================================================
    # GHOST TRADE MANAGEMENT
    # ======================================================================
    ghost = asset_state.ghost_trade
    if ghost:
        ghost_done = False
        if ghost['type'] == 'LONG':
            if low <= ghost['sl']:
                ghost_done = True
            else:
                if not ghost['partial_closed']:
                    tp1_price = ghost['entry'] + (ghost['risk'] * ghost['tp1_rr'])
                    if high >= tp1_price:
                        ghost['partial_closed'] = True
                        ghost['sl'] = ghost['entry']
                if high >= ghost['tp']:
                    ghost_done = True
        else:  # SHORT
            if high >= ghost['sl']:
                ghost_done = True
            else:
                if not ghost['partial_closed']:
                    tp1_price = ghost['entry'] - (ghost['risk'] * ghost['tp1_rr'])
                    if low <= tp1_price:
                        ghost['partial_closed'] = True
                        ghost['sl'] = ghost['entry']
                if low <= ghost['tp']:
                    ghost_done = True
        if ghost_done:
            logger.info(f"[{symbol}] Ghost trade ended ({ghost['type']})")
            asset_state.ghost_trade = None
        else:
            result['event'] = "GHOST_ACTIVE"
            result['event_details'] = f"{ghost['type']} entry={ghost['entry']:.2f} sl={ghost['sl']:.2f} tp={ghost['tp']:.2f}"
            result['state_after'] = asset_state.state
            return result

    # Déterminer la session
    curr_sess = get_session(candle_dt)
    result['session'] = curr_sess
    asset_sessions = config.get('sessions', {})

    # Hors session -> INSIDE
    if not asset_sessions.get(curr_sess, False):
        asset_state.state = "INSIDE"
        asset_state.breakout_direction = None
        asset_state.candles_since_breakout = 0
        result['state_after'] = "INSIDE"
        result['event'] = "OUT_OF_SESSION"
        return result

    # Reset VP au debut de session
    if RESET_VP_PER_SESSION:
        session_start = is_session_start(candle_dt)
        if session_start and asset_sessions.get(session_start, False):
            asset_state.reset_vp()
            asset_state.current_session = session_start
            result['event'] = "SESSION_START"
            result['event_details'] = session_start

    # Construire VP depuis debut de session jusqu'à FIN de la bougie actuelle
    session_cfg = SESSIONS_CONFIG.get(curr_sess, {})
    session_start_hour = session_cfg.get('vp_start', 0)
    session_start_dt = candle_dt.replace(hour=int(session_start_hour), minute=int((session_start_hour % 1) * 60), second=0, microsecond=0)
    if session_start_dt > candle_dt:
        session_start_dt -= timedelta(days=1)

    end_dt = candle_dt + timedelta(minutes=1)

    prices, volumes = fetch_ticks_from_db(engine, config['tick_table'], session_start_dt, end_dt)

    if len(prices) > 0:
        asset_state.vp.reset()
        asset_state.vp.add_ticks(prices, volumes)

    # Update structural levels tracker with ticks for current minute
    tick_start = candle_dt
    tick_end = candle_dt + timedelta(minutes=1)
    tick_prices, tick_volumes = fetch_ticks_from_db(engine, config['tick_table'], tick_start, tick_end)
    asset_state.structural.update(candle_dt, tick_prices, tick_volumes)

    poc, vah, val = asset_state.vp.get_levels()
    if poc is None:
        result['event'] = "NO_VP"
        result['event_details'] = f"0 ticks from {session_start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')}"
        result['state_after'] = asset_state.state
        return result

    poc_strength = asset_state.vp.get_poc_strength()
    vp_shape = asset_state.vp.get_profile_shape()

    result['poc'] = poc
    result['vah'] = vah
    result['val'] = val
    result['poc_strength'] = poc_strength
    result['vp_shape'] = vp_shape

    state = asset_state.state

    # ======================================================================
    # STATE MACHINE — Combined MR + CB
    # ======================================================================

    if state == "INSIDE":
        if close > vah:
            asset_state.state = "BREAKOUT_UP"
            asset_state.breakout_direction = "UP"
            asset_state.breakout_time = candle_dt
            asset_state.candles_since_breakout = 1
            asset_state.wait_highs = [high]
            asset_state.wait_lows = [low]
            result['event'] = "BREAKOUT_UP"
        elif close < val:
            asset_state.state = "BREAKOUT_DOWN"
            asset_state.breakout_direction = "DOWN"
            asset_state.breakout_time = candle_dt
            asset_state.candles_since_breakout = 1
            asset_state.wait_highs = [high]
            asset_state.wait_lows = [low]
            result['event'] = "BREAKOUT_DOWN"
        else:
            result['event'] = "INSIDE"

    elif state in ("BREAKOUT_UP", "BREAKOUT_DOWN"):
        asset_state.candles_since_breakout += 1
        asset_state.wait_highs.append(high)
        asset_state.wait_lows.append(low)

        breakout_dir = asset_state.breakout_direction
        reintegrated = False
        confirmed_outside = False

        # Check reintegration (strict: close INSIDE VA)
        if ENABLE_MR:
            if breakout_dir == "UP" and close < vah:
                reintegrated = True
            elif breakout_dir == "DOWN" and close > val:
                reintegrated = True
        else:
            if breakout_dir == "UP" and close <= vah:
                reintegrated = True
            elif breakout_dir == "DOWN" and close >= val:
                reintegrated = True

        # Check confirmed breakout
        if not reintegrated and asset_state.candles_since_breakout >= WAIT_CANDLES:
            confirmed_outside = True

        # ==============================================================
        # MR ATTEMPT (reintegrated)
        # ==============================================================
        if reintegrated and ENABLE_MR:
            direction = 'SHORT' if breakout_dir == "UP" else 'LONG'

            can_direction = (direction == 'LONG' and config['allow_long']) or (direction == 'SHORT' and config['allow_short'])
            can_time = can_trade_now(candle_dt, curr_sess)
            day_ok = candle_dt.weekday() in config.get('allowed_days', [0, 1, 2, 3, 4])
            hour_ok = candle_dt.hour not in MR_EXCLUDED_HOURS

            mr_poc_ok = True
            if poc_strength is None or poc_strength < MR_MIN_POC_STRENGTH:
                mr_poc_ok = False
                result['event'] = "MR_FILTERED_POC_STRENGTH"
                result['event_details'] = f"{poc_strength:.2f}x < {MR_MIN_POC_STRENGTH}x"

            mr_duration_ok = True
            breakout_duration_min = (candle_dt - asset_state.breakout_time).total_seconds() / 60.0
            if breakout_duration_min >= MR_MAX_BREAKOUT_DURATION_MINUTES:
                mr_duration_ok = False
                if mr_poc_ok:
                    result['event'] = "MR_FILTERED_DURATION"
                    result['event_details'] = f"{breakout_duration_min:.1f}min >= {MR_MAX_BREAKOUT_DURATION_MINUTES}min"

            shape_ok = True
            if USE_VP_SHAPE_FILTER and vp_shape in EXCLUDED_VP_SHAPES:
                shape_ok = False

            if mr_poc_ok and mr_duration_ok and shape_ok and can_direction and can_time and day_ok and hour_ok:
                if direction == 'SHORT':
                    swing_high = max(asset_state.wait_highs)
                    sl = swing_high + MR_SL_OFFSET
                    risk = sl - close
                else:
                    swing_low = min(asset_state.wait_lows)
                    sl = swing_low - MR_SL_OFFSET
                    risk = close - sl

                if risk > 0:
                    tp = poc
                    if direction == 'SHORT':
                        actual_rr = (close - tp) / risk
                    else:
                        actual_rr = (tp - close) / risk

                    poc_filter_ok = True
                    if MR_FILTER_ENTRY_VS_POC:
                        if direction == 'SHORT' and close < poc:
                            poc_filter_ok = False
                        elif direction == 'LONG' and close > poc:
                            poc_filter_ok = False

                    rr_ok = MR_MIN_RR <= actual_rr <= 30
                    tp_in_va = True
                    if direction == 'SHORT' and tp < val:
                        tp_in_va = False
                    if direction == 'LONG' and tp > vah:
                        tp_in_va = False

                    if rr_ok and poc_filter_ok and tp_in_va:
                        if MR_USE_TRAILING:
                            if direction == 'LONG':
                                tp1 = close + (risk * MR_TP1_RR)
                            else:
                                tp1 = close - (risk * MR_TP1_RR)
                        else:
                            tp1 = tp

                        tp2 = tp
                        risk_amount = account_balance * RISK_PERCENT

                        order_type = "BUY" if direction == 'LONG' else "SELL"
                        result['event'] = f"MR_TRADE_{direction}"
                        result['event_details'] = f"Entry={close} SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f} RR={actual_rr:.2f}"

                        success = place_market_order(
                            mt5_symbol, order_type, close, sl, tp1, tp2,
                            risk_amount, config['tick_size'],
                            tp1_split=MR_TP1_SPLIT, tp2_split=MR_TP2_SPLIT,
                            strategy="VP_MR"
                        )
                        if success:
                            asset_state.last_trade_candle_ts = candle_ts
                    else:
                        if result['event'] is None:
                            reasons = []
                            if not rr_ok: reasons.append(f"RR={actual_rr:.2f}")
                            if not poc_filter_ok: reasons.append("entry_vs_poc")
                            if not tp_in_va: reasons.append("tp_outside_va")
                            result['event'] = "MR_FILTERED_ENTRY"
                            result['event_details'] = ", ".join(reasons)
            elif can_direction and not can_time:
                if result['event'] is None:
                    result['event'] = "MR_FILTERED_TRADE_HOURS"

            # Reset after MR attempt
            asset_state.state = "INSIDE"
            asset_state.breakout_direction = None
            asset_state.candles_since_breakout = 0
            asset_state.wait_highs = []
            asset_state.wait_lows = []

        # ==============================================================
        # CB ATTEMPT (confirmed outside)
        # ==============================================================
        elif confirmed_outside and ENABLE_CB:
            direction = 'LONG' if breakout_dir == "UP" else 'SHORT'

            can_direction = (direction == 'LONG' and config['allow_long']) or (direction == 'SHORT' and config['allow_short'])
            can_time = can_trade_now(candle_dt, curr_sess)
            day_ok = candle_dt.weekday() in config.get('allowed_days', [0, 1, 2, 3, 4])
            hour_ok = candle_dt.hour not in CB_EXCLUDED_HOURS

            cb_poc_ok = True
            if poc_strength is None or poc_strength < CB_MIN_POC_STRENGTH:
                cb_poc_ok = False
                result['event'] = "CB_FILTERED_POC_STRENGTH"
                result['event_details'] = f"{poc_strength:.2f}x < {CB_MIN_POC_STRENGTH}x"

            shape_ok = True
            if USE_VP_SHAPE_FILTER and vp_shape in EXCLUDED_VP_SHAPES:
                shape_ok = False

            if cb_poc_ok and shape_ok and can_direction and can_time and day_ok and hour_ok:
                if direction == 'LONG':
                    swing_low = min(asset_state.wait_lows)
                    sl = swing_low - CB_SL_OFFSET
                    risk = close - sl
                else:
                    swing_high = max(asset_state.wait_highs)
                    sl = swing_high + CB_SL_OFFSET
                    risk = sl - close

                if risk > 0:
                    # Structural target
                    targets = asset_state.structural.get_target_levels(direction, close)

                    tp = None
                    tp_label = None
                    for label, price in targets:
                        if direction == 'LONG':
                            rr = (price - close) / risk
                        else:
                            rr = (close - price) / risk
                        if CB_MIN_RR <= rr <= 30:
                            tp = price
                            tp_label = label
                            break

                    # Ghost trade: PD_POC target → block slot without risking capital
                    if tp is not None and tp_label == 'PD_POC':
                        asset_state.ghost_trade = {
                            'type': direction, 'entry': close, 'sl': sl, 'tp': tp,
                            'tp1_rr': CB_TP1_RR, 'risk': risk,
                            'partial_closed': False,
                        }
                        logger.info(f"[{symbol}] Ghost trade started: {direction} entry={close:.2f} sl={sl:.2f} tp={tp:.2f} (PD_POC)")
                        result['event'] = "CB_GHOST_TRADE"
                        result['event_details'] = f"{direction} → PD_POC={tp:.2f}"
                        asset_state.state = "INSIDE"
                        asset_state.breakout_direction = None
                        asset_state.candles_since_breakout = 0
                        asset_state.wait_highs = []
                        asset_state.wait_lows = []
                        result['state_after'] = asset_state.state
                        return result

                    if tp is None:
                        # Fallback: fixed RR
                        if direction == 'LONG':
                            tp = close + (risk * CB_MIN_RR)
                        else:
                            tp = close - (risk * CB_MIN_RR)
                        tp_label = f"FIXED_{CB_MIN_RR}R"
                        actual_rr = CB_MIN_RR
                    else:
                        if direction == 'LONG':
                            actual_rr = (tp - close) / risk
                        else:
                            actual_rr = (close - tp) / risk

                    if CB_USE_TRAILING:
                        if direction == 'LONG':
                            tp1 = close + (risk * CB_TP1_RR)
                        else:
                            tp1 = close - (risk * CB_TP1_RR)
                    else:
                        tp1 = tp

                    tp2 = tp
                    risk_amount = account_balance * RISK_PERCENT

                    order_type = "BUY" if direction == 'LONG' else "SELL"
                    result['event'] = f"CB_TRADE_{direction}"
                    result['event_details'] = f"Entry={close} SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f} RR={actual_rr:.2f} Target={tp_label}"

                    success = place_market_order(
                        mt5_symbol, order_type, close, sl, tp1, tp2,
                        risk_amount, config['tick_size'],
                        tp1_split=CB_TP1_SPLIT, tp2_split=CB_TP2_SPLIT,
                        strategy="VP_CB"
                    )
                    if success:
                        asset_state.last_trade_candle_ts = candle_ts

            elif can_direction and not can_time:
                if result['event'] is None:
                    result['event'] = "CB_FILTERED_TRADE_HOURS"

            # Reset after CB attempt
            asset_state.state = "INSIDE"
            asset_state.breakout_direction = None
            asset_state.candles_since_breakout = 0
            asset_state.wait_highs = []
            asset_state.wait_lows = []

        elif reintegrated and not ENABLE_MR:
            # MR disabled, reintegration just resets
            asset_state.state = "INSIDE"
            asset_state.breakout_direction = None
            asset_state.candles_since_breakout = 0
            asset_state.wait_highs = []
            asset_state.wait_lows = []

        elif confirmed_outside and not ENABLE_CB:
            # CB disabled, stay in BREAKOUT state waiting for reintegration
            pass

        else:
            # Still waiting (not enough candles, not reintegrated)
            result['event'] = f"STILL_{state}"
            result['event_details'] = f"candles={asset_state.candles_since_breakout}/{WAIT_CANDLES}"

    result['state_after'] = asset_state.state
    return result


def log_candle_info(result: dict, balance: float, equity: float):
    """Log les infos importantes"""
    if not result['new_candle']:
        return

    event = result['event'] or "NONE"

    if event in ["INSIDE", "OUT_OF_SESSION", "STILL_BREAKOUT_UP", "STILL_BREAKOUT_DOWN"]:
        return

    symbol = result['symbol']
    dt = result['candle_dt'].strftime('%Y-%m-%d %H:%M') if result['candle_dt'] else "N/A"
    session = result['session'] or "N/A"

    logger.info(f"{'─' * 80}")
    logger.info(f"[{symbol}] {dt} UTC | Session: {session} | Balance: ${balance:,.2f} | Equity: ${equity:,.2f}")

    if result['close']:
        logger.info(f"[{symbol}] Price: Close={result['close']:.2f} | High={result['high']:.2f} | Low={result['low']:.2f}")

    if result['poc']:
        logger.info(f"[{symbol}] VP: POC={result['poc']:.2f} | VAH={result['vah']:.2f} | VAL={result['val']:.2f} | Strength={result['poc_strength']:.2f}x | Shape={result['vp_shape']}")

    state_change = ""
    if result['state_before'] != result['state_after']:
        state_change = f" -> {result['state_after']}"
    logger.info(f"[{symbol}] State: {result['state_before']}{state_change}")

    details = result['event_details'] or ""

    if "TRADE" in event:
        tag = "[TRADE]"
    elif "GHOST" in event:
        tag = "[GHOST]"
    elif event.startswith("BREAKOUT"):
        tag = "[BREAKOUT]"
    elif "FILTERED" in event:
        tag = "[FILTERED]"
    elif event == "SESSION_START":
        tag = "[SESSION]"
    elif event == "POSITION_ACTIVE":
        tag = "[POSITION]"
    elif event == "GHOST_ACTIVE":
        tag = "[GHOST]"
    elif event == "NO_VP":
        tag = "[WARN]"
    else:
        tag = "[INFO]"

    if details:
        logger.info(f"[{symbol}] {tag} {event} | {details}")
    else:
        logger.info(f"[{symbol}] {tag} {event}")


# =============================================================================
# MAIN
# =============================================================================

def warmup_structural_levels(asset_states: dict, engine):
    """Load historical ticks to populate prev_day and prev_week structural levels."""
    now = datetime.now(timezone.utc)

    for symbol, asset_state in asset_states.items():
        config = asset_state.config
        tick_table = config['tick_table']

        # Load last week's ticks for prev_week
        week_start = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)

        logger.info(f"[{symbol}] Loading prev_week ticks: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
        prices, volumes = fetch_ticks_from_db(engine, tick_table, week_start, week_end)
        if len(prices) > 0:
            # Feed day by day to trigger day boundaries
            tick_df = pd.DataFrame({'price': prices, 'volume': volumes})
            # Use a single update with a representative date from last week
            mid_week = week_start + timedelta(days=2)
            asset_state.structural.update(mid_week, prices, volumes)
            logger.info(f"[{symbol}] Loaded {len(prices)} prev_week ticks")

        # Load yesterday's ticks for prev_day
        yesterday_start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

        logger.info(f"[{symbol}] Loading prev_day ticks: {yesterday_start.strftime('%Y-%m-%d')}")
        prices, volumes = fetch_ticks_from_db(engine, tick_table, yesterday_start, yesterday_end)
        if len(prices) > 0:
            asset_state.structural.update(yesterday_start, prices, volumes)
            logger.info(f"[{symbol}] Loaded {len(prices)} prev_day ticks")

        # Trigger day boundary by feeding a tick from today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = now
        prices, volumes = fetch_ticks_from_db(engine, tick_table, today_start, today_end)
        if len(prices) > 0:
            asset_state.structural.update(now, prices, volumes)

        pd = asset_state.structural.prev_day
        pw = asset_state.structural.prev_week
        logger.info(f"[{symbol}] Structural prev_day: POC={pd['poc']} VAH={pd['vah']} VAL={pd['val']}")
        logger.info(f"[{symbol}] Structural prev_week: POC={pw['poc']} VAH={pw['vah']} VAL={pw['val']}")


def main():
    logger.info("=" * 60)
    logger.info("Combined MR + CB - Live Trading")
    logger.info("ALIGNED ON backtest_combined_mr_breakout")
    logger.info(f"MR={'ON' if ENABLE_MR else 'OFF'} | CB={'ON' if ENABLE_CB else 'OFF'} | WAIT_CANDLES={WAIT_CANDLES}")
    logger.info("=" * 60)

    if not mt5.initialize():
        logger.error("MT5 initialization failed")
        return

    account_info = mt5.account_info()
    if account_info is None:
        logger.error("Failed to get account info")
        mt5.shutdown()
        return

    logger.info(f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}")

    engine = get_pg_engine()

    enabled_assets = [a for a in ASSETS if a.get('enabled', False)]
    if not enabled_assets:
        logger.error("No enabled assets")
        mt5.shutdown()
        return

    asset_states = {}
    for config in enabled_assets:
        asset_states[config['symbol']] = AssetState(config)
        sessions = [s for s, v in config.get('sessions', {}).items() if v]
        logger.info(f"Asset: {config['symbol']} | Sessions: {sessions}")

    load_all_states(asset_states)

    # Warmup structural levels (prev_day, prev_week)
    warmup_structural_levels(asset_states, engine)

    # Afficher la config des sessions
    logger.info("Sessions config (VP hours / Trade hours):")
    for sess_name, cfg in SESSIONS_CONFIG.items():
        logger.info(f"  {sess_name}: VP={cfg['vp_start']}-{cfg['vp_end']} | Trade={cfg['trade_start']}-{cfg['trade_end']}")

    logger.info(f"MR: RR>={MR_MIN_RR} | SL_OFF={MR_SL_OFFSET} | POC>={MR_MIN_POC_STRENGTH}x | Duration<{MR_MAX_BREAKOUT_DURATION_MINUTES}min | Split={MR_TP1_SPLIT:.0%}/{MR_TP2_SPLIT:.0%}")
    logger.info(f"CB: RR>={CB_MIN_RR} | SL_OFF={CB_SL_OFFSET} | POC>={CB_MIN_POC_STRENGTH}x | ExclHours={CB_EXCLUDED_HOURS} | Split={CB_TP1_SPLIT:.0%}/{CB_TP2_SPLIT:.0%}")
    logger.info(f"Risk: {RISK_PERCENT*100}%")
    logger.info("=" * 60)

    try:
        while True:
            ref_asset = list(asset_states.values())[0] if asset_states else None
            db_time = None
            if ref_asset:
                ref_candles = fetch_candles_from_db(engine, ref_asset.config['candle_table'], limit=1)
                if ref_candles:
                    db_time = ref_candles[0]['dt']
            
            if db_time is None:
                logger.warning("No DB data available, waiting...")
                time.sleep(LOOP_INTERVAL_SECONDS)
                continue
            
            if daily_market_close_guard(db_time):
                time.sleep(60)
                continue
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Lost connection to MT5")
                time.sleep(10)
                continue
            
            balance = account_info.balance
            equity = account_info.equity
            
            for symbol, asset_state in asset_states.items():
                try:
                    manage_tp1_to_be(asset_state.mt5_symbol, asset_state.config['tick_size'])
                    result = detect_and_trade(asset_state, engine, balance)
                    log_candle_info(result, balance, equity)
                except Exception as e:
                    logger.error(f"[{symbol}] Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            save_all_states(asset_states)
            time.sleep(LOOP_INTERVAL_SECONDS)
    
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        mt5.shutdown()
        logger.info("MT5 shutdown complete")


if __name__ == "__main__":
    main()