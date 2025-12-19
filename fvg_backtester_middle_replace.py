#!/usr/bin/env python3
# postgres_fvg_backtester_REPLACE_LOGIC_CORRECTED.py

import os, re, csv, sys, time
import argparse
import statistics
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional, Tuple, Dict, List, Any

# Outils de Base de Données
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table, Column, BigInteger, String, select, desc, inspect, and_, text
from sqlalchemy.types import Numeric

UTC = timezone.utc
DATE_FORMAT = "%Y-%m-%d"

# ---------- CONFIG DE TRADING ----------
DEFAULT_RR = Decimal("1.0")
MAX_WAIT_CANDLES = 4
SCAN_TF = "30m" 
EXECUTION_TF_SUFFIX = "1m" 
INITIAL_BALANCE = Decimal("10000.00")
DEFAULT_RISK_PER_TRADE = Decimal("0.01") 

# ---------- CONSTANTES POUR STDEV ----------
STDEV_PERIOD = 200 
DEFAULT_STDEV_THRESHOLD = 0.5
DEFAULT_STDEV_MAX = 1.0
# ----------------------------------------

# --- Structure pour les résultats globaux ---
GLOBAL_RESULTS = []

# ---------- UTILS BDD & GENERALES (Inchangées) ----------

def price_scale(base: str, quote: str) -> int:
    return 3 if ("JPY" in (base, quote)) else 5

def qround(x: float | Decimal, scale: int) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("1").scaleb(-scale), rounding=ROUND_HALF_UP)

def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=UTC).strftime('%m-%d %H:%M')

def parse_date_to_ms(date_str: str, is_end_date: bool = False) -> int:
    try:
        dt = datetime.strptime(date_str, DATE_FORMAT).replace(tzinfo=UTC)
        if is_end_date:
            dt += timedelta(days=1) - timedelta(milliseconds=1)
        return int(dt.timestamp() * 1000)
    except ValueError:
        raise ValueError(f"Le format de date doit être {DATE_FORMAT} (Ex: 2024-01-15)")

def parse_pairs(path: str):
    out = []
    if not os.path.exists(path):
        print(f"[WARN] Fichier de paires non trouvé: {path}")
        return []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = r.get("pair") or r.get("PAIR") or r.get("Pair")
            if p:
                out.append(p.strip())
    return out

def sanitize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def get_pg_engine():
    load_dotenv()
    host = os.getenv("PG_HOST", "127.0.0.1")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB", "postgres")
    user = os.getenv("PG_USER", "postgres")
    pwd = os.getenv("PG_PASSWORD", "postgres")
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode=disable",
            pool_pre_ping=True,
            future=True
        )
        return engine
    except Exception as e:
        print(f"[FATAL] Échec de la connexion PostgreSQL: {e}")
        sys.exit(1)


# --- Fetch Rates ---

def fetch_all_rates(engine, pair: str, tf: str, start_ms: Optional[int], end_ms: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    base, quote = pair[:3], pair[3:]
    scale = price_scale(base, quote)
    table_name = f"candles_mt5_{sanitize_name(pair)}_{sanitize_name(tf)}"

    meta = MetaData()
    table = Table(
        table_name, meta,
        Column("ts", BigInteger, primary_key=True),
        Column("open", Numeric(20, scale)),
        Column("high", Numeric(20, scale)),
        Column("low", Numeric(20, scale)),
        Column("close", Numeric(20, scale)),
        Column("ema_50", Numeric(20, scale)), 
    )
    
    inspector = inspect(engine) 
    if not inspector.has_table(table_name):
        return None

    try:
        with engine.connect() as conn:
            conditions = []
            if start_ms is not None:
                safe_buffer_ms = timedelta(days=20).total_seconds() * 1000
                conditions.append(table.c.ts >= (start_ms - safe_buffer_ms)) 
            if end_ms is not None:
                conditions.append(table.c.ts <= end_ms)
            
            q = select(
                table.c.ts.label("time"), 
                table.c.open, 
                table.c.high, 
                table.c.low, 
                table.c.close,
                table.c.ema_50
            ).where(and_(*conditions)).order_by(table.c.ts.asc())
            
            rows = conn.execute(q).fetchall()
        
        if not rows:
            return None

        rates = []
        for row in rows: 
            rates.append({
                "time": int(row.time), 
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "ema_50": float(row.ema_50) if row.ema_50 is not None else None
            })

        return rates
        
    except Exception as e:
        print(f"[ERR] BDD error for {pair}/{tf}: {e}")
        return None


# --- SIMULATION M1 PAR BLOC ---

class SimulationState:
    def __init__(self):
        self.pending_order = None # {entry, sl, tp, side, expiration, setup_data}
        self.active_trade = None  # {entry, sl, tp, side, setup_data}
        self.closed_trade = None  # {result, exit_time, pnl_r...}

def process_m1_chunk(engine, pair: str, start_ts: int, end_ts: int, state: SimulationState):
    """
    Simule ce qui se passe sur le marché M1 entre deux bougies 30m.
    Met à jour l'état (Pending -> Active, Active -> Closed, Pending -> Expired).
    """
    table_m1 = f"candles_mt5_{sanitize_name(pair)}_{EXECUTION_TF_SUFFIX}"
    
    # On récupère les M1 pour cette tranche horaire
    sql = text(f"SELECT ts, high, low FROM {table_m1} WHERE ts >= :start AND ts < :end ORDER BY ts ASC")
    
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"start": start_ts, "end": end_ts}).fetchall()
    except Exception:
        return 

    if not rows: return

    for row in rows:
        ts = int(row.ts)
        high = float(row.high)
        low = float(row.low)

        # 1. GESTION TRADE ACTIF (Prioritaire)
        if state.active_trade:
            t = state.active_trade
            res = None
            
            if t['side'] == "LONG":
                if low <= t['sl']: res = "LOSS"
                elif high >= t['tp']: res = "WIN"
            else: # SHORT
                if high >= t['sl']: res = "LOSS"
                elif low <= t['tp']: res = "WIN"
            
            if res:
                # Trade terminé
                state.closed_trade = {
                    "pair": pair, "entry_time": t['setup_data']['time'], "exit_time": ts,
                    "side": t['side'], "entry_price": t['entry'], "sl_price": t['sl'],
                    "tp_price": t['tp'], "result": res, 
                    "exit_price": t['sl'] if res == "LOSS" else t['tp']
                }
                state.active_trade = None
                return # On arrête le chunk ici, le trade est fini

        # 2. GESTION ORDRE EN ATTENTE (Pending)
        elif state.pending_order:
            p = state.pending_order
            
            # A. Vérification Expiration
            if ts > p['expiration']:
                state.pending_order = None # Expiré
                continue # Passe à la minute suivante

            # B. Vérification Trigger (Entrée)
            triggered = False
            if p['side'] == "LONG":
                if low <= p['entry']: triggered = True
            else: # SHORT
                if high >= p['entry']: triggered = True
            
            if triggered:
                # L'ordre devient actif
                state.active_trade = {
                    "entry": p['entry'], "sl": p['sl'], "tp": p['tp'], 
                    "side": p['side'], "setup_data": p['setup_data']
                }
                state.pending_order = None
                
                # Vérification immédiate du SL/TP sur la MÊME minute (volatilité)
                t = state.active_trade
                res = None
                if t['side'] == "LONG":
                    if low <= t['sl']: res = "LOSS" # Pire cas: SL touché
                else:
                    if high >= t['sl']: res = "LOSS"
                
                if res:
                    state.closed_trade = {
                        "pair": pair, "entry_time": t['setup_data']['time'], "exit_time": ts,
                        "side": t['side'], "entry_price": t['entry'], "sl_price": t['sl'],
                        "tp_price": t['tp'], "result": res, 
                        "exit_price": t['sl']
                    }
                    state.active_trade = None
                    return


# --- FVG Volatility Check ---

def check_fvg_volatility(rates: List[Dict[str, Any]], i: int, threshold: float) -> Tuple[bool, bool, float, float]:
    if i < STDEV_PERIOD + 2: return False, False, 0.0, 0.0
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i = rates[i]["high"]; l_i = rates[i]["low"]
    raw_bull_cond = (h_i_2 < l_i); raw_bear_cond = (l_i_2 > h_i)
    if not raw_bull_cond and not raw_bear_cond: return False, False, 0.0, 0.0
    subset = rates[i - STDEV_PERIOD - 2: i + 1]
    lows = [r["low"] for r in subset]; highs = [r["high"] for r in subset]
    diffs = []
    for k in range(2, len(lows)):
        gap = abs(lows[k] - highs[k-2])
        diffs.append(gap)
    recent_diffs = diffs[-STDEV_PERIOD:]
    try:
        volatility = statistics.stdev(recent_diffs)
    except statistics.StatisticsError: return False, False, 0.0, 0.0
    if volatility == 0: volatility = 1.0e-9 
    
    is_bullish = False; is_bearish = False; score = 0.0; current_gap = 0.0
    if raw_bull_cond:
        current_gap = l_i - h_i_2; score = current_gap / volatility
        if score > threshold: is_bullish = True
    elif raw_bear_cond:
        current_gap = l_i_2 - h_i; score = current_gap / volatility
        if score > threshold: is_bearish = True
    return is_bullish, is_bearish, score, current_gap

def detect_fvg_setup(rates: List[Dict[str, Any]], i: int, scale: int, stdev_threshold: float, stdev_max: float) -> Optional[Dict[str, Any]]:
    if i < 2: return None
    ema50 = rates[i]["ema_50"]
    h_i_2 = rates[i-2]["high"]; l_i_2 = rates[i-2]["low"]
    h_i_1 = rates[i-1]["high"]; l_i_1 = rates[i-1]["low"] 
    
    if ema50 is None: return None
    
    is_bull_stdev, is_bear_stdev, score, current_gap = check_fvg_volatility(rates, i, stdev_threshold)
    if not is_bull_stdev and not is_bear_stdev: return None
    if score > stdev_max: return None

    ema_ok = False
    entry_price = Decimal(0); sl_price = Decimal(0)
    
    if is_bull_stdev:
        entry_side = "LONG"
        fvg_high = Decimal(str(rates[i]["low"])); fvg_low = Decimal(str(h_i_2))
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        sl_price = Decimal(str(l_i_1))
        ema_ok = entry_price > Decimal(str(ema50))
        if sl_price >= entry_price: return None

    elif is_bear_stdev:
        entry_side = "SHORT"
        fvg_high = Decimal(str(l_i_2)); fvg_low = Decimal(str(rates[i]["high"]))
        entry_price = (fvg_high + fvg_low) / Decimal("2.0")
        sl_price = Decimal(str(h_i_1))
        ema_ok = entry_price < Decimal(str(ema50))
        if sl_price <= entry_price: return None
             
    else: return None  

    if not ema_ok: return None
    
    return {
        "side": entry_side,
        "entry_price": qround(entry_price, scale),  
        "sl_price": qround(sl_price, scale),
        "fvg_start_candle_index": i,
        "stdev_score": score,
        "time": rates[i]["time"]
    }


# ---------- LOGIQUE DE BACKTESTING (CORRIGÉE : DETECTION D'ABORD) ----------

def execute_backtest(engine, rates: List[Dict[str, Any]], pair: str, rr_ratio: Decimal, scale: int, stdev_threshold: float, start_ms: Optional[int], end_ms: Optional[int], risk_per_trade: Decimal, stdev_max: float) -> List[Dict[str, Any]]:
    
    if len(rates) < 200: return []
    
    # 1. Déterminer la plage d'itération
    start_index = 0
    end_index = len(rates)
    required_seed = STDEV_PERIOD + 2 
    
    if start_ms is not None:
        for idx in range(required_seed, len(rates)):
            if rates[idx]['time'] >= start_ms:
                start_index = idx; break
        else: return []

    if end_ms is not None:
        for idx in range(start_index, len(rates)):
            if rates[idx]['time'] > end_ms:
                end_index = idx; break

    if end_index <= start_index: return []
    
    # Initialisation Stats
    balance_r = Decimal(0); total_trades = 0; wins = 0; losses = 0
    peak_r = Decimal(0); max_drawdown_r = Decimal(0)
    trade_log: List[Dict[str, Any]] = []
    
    summary_start_ts = rates[start_index]['time']
    summary_end_ts = rates[end_index - 1]['time']
    
    scan_duration_ms = 0
    if len(rates) > 1: scan_duration_ms = rates[1]['time'] - rates[0]['time']
    if scan_duration_ms == 0: scan_duration_ms = 300000 
    
    max_wait_ms = MAX_WAIT_CANDLES * scan_duration_ms

    # --- ÉTAT DU SIMULATEUR ---
    sim_state = SimulationState()

    # 2. Itération pas à pas (Bougie par Bougie)
    for i in range(start_index, end_index):
        
        current_ts = rates[i]['time']
        
        # --- ETAPE 1 : DÉTECTION NOUVEAU SIGNAL (Sur la bougie 'i' qui vient de clore) ---
        setup = detect_fvg_setup(rates, i, scale, stdev_threshold, stdev_max)
        
        # On définit le temps de début de la PROCHAINE bougie (où l'ordre sera actif)
        next_candle_start_m1 = current_ts + scan_duration_ms
        next_candle_end_m1 = current_ts + 2 * scan_duration_ms 

        if setup:
            # Si un trade est DÉJÀ en cours (Active), on ne fait rien (on ignore le setup)
            if sim_state.active_trade:
                pass
            else:
                # Si pas de trade actif (soit rien, soit un Pending existant)
                # ON ÉCRASE (Remplace) le Pending existant par le nouveau
                
                stop_loss_risk = abs(setup["entry_price"] - setup["sl_price"])
                if setup["side"] == "LONG": target_price = setup["entry_price"] + stop_loss_risk * rr_ratio
                else: target_price = setup["entry_price"] - stop_loss_risk * rr_ratio
                tp_price = qround(target_price, scale)
                
                sim_state.pending_order = {
                    "entry": float(setup["entry_price"]),
                    "sl": float(setup["sl_price"]),
                    "tp": float(tp_price),
                    "side": setup["side"],
                    # L'expiration est calculée à partir de maintenant
                    "expiration": next_candle_start_m1 + max_wait_ms,
                    "setup_data": setup
                }
        
        # --- ETAPE 2 : SIMULATION DE LA PÉRIODE SUIVANTE (i -> i+1) ---
        # On regarde si l'ordre (le nouveau ou l'ancien) se déclenche dans les 30 prochaines minutes
        
        process_m1_chunk(engine, pair, next_candle_start_m1, next_candle_end_m1, sim_state)
        
        # Si un trade s'est terminé durant ce chunk
        if sim_state.closed_trade:
            ct = sim_state.closed_trade
            pnl_r = rr_ratio if ct['result'] == "WIN" else Decimal("-1.0")
            
            total_trades += 1
            if ct['result'] == "WIN": wins += 1
            else: losses += 1
            
            balance_r += pnl_r
            peak_r = max(peak_r, balance_r)
            drawdown_r = peak_r - balance_r
            max_drawdown_r = max(max_drawdown_r, drawdown_r)
            
            ct['pnl_r'] = pnl_r
            trade_log.append(ct)
            sim_state.closed_trade = None # Reset pour prêt à recevoir prochain trade
            
    
    # --- FIN BOUCLE ---
    
    # Calcul stats finales
    expectancy_r = balance_r / total_trades if total_trades > 0 else Decimal(0)
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    max_drawdown_percent = max_drawdown_r * risk_per_trade * Decimal("100") 
    
    GLOBAL_RESULTS.append({
        "pair": pair, "total_trades": total_trades, "wins": wins, "losses": losses,
        "expectancy_r": expectancy_r, "max_drawdown_r": max_drawdown_r,
        "max_drawdown_percent": max_drawdown_percent, "win_rate": win_rate,  
        "net_profit_euro": 0, "final_balance": 0, 
        "start_ts": summary_start_ts, "end_ts": summary_end_ts
    })
    
    return trade_log

# ---------- FONCTIONS D'AFFICHAGE (Inchangées) ----------

def display_summary_table(rr_ratio: Decimal, stdev_threshold: float, risk_perc: Decimal, stdev_max: float, results: List[Dict[str, Any]]):
    results.sort(key=lambda x: x['expectancy_r'], reverse=True)
    total_trades_all = sum(res['total_trades'] for res in results)
    total_wins_all = sum(res['wins'] for res in results)
    total_losses_all = sum(res['losses'] for res in results)
    global_win_rate = (total_wins_all / total_trades_all) * 100 if total_trades_all > 0 else 0.0
    total_expectancy = sum(res['expectancy_r'] * res['total_trades'] for res in results if res['total_trades'] > 0)
    weighted_expectancy = total_expectancy / total_trades_all if total_trades_all > 0 else Decimal(0)
    
    print("\n" + "="*105)
    print(f"SUMMARY BACKTEST FVG/EMA 50 (TF: {SCAN_TF}, RR: {rr_ratio}R, RISK: {risk_perc*Decimal(100)}%)")
    print(f"MODE: REPLACE PENDING (Corrected Order)")
    print("="*105)
    
    header = "| {:^10} | {:^8} | {:^8} | {:^10} | {:^12} | {:^12} | {:^12} |".format("PAIRE", "TRADES", "GAINS", "PERTES", "WIN RATE", "EXPECTANCY", "MAX DD (%)")
    separator = "|" + "-"*12 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*14 + "|" + "-"*14 + "|"
    print(header); print(separator)
    
    for res in results:
        print("| {:<10} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12} | {:>12} |".format(
            res['pair'], res['total_trades'], res['wins'], res['losses'], 
            f"{res['win_rate']:.2f}%", f"{float(res['expectancy_r']):.4f}R", f"{float(res['max_drawdown_percent']):.2f}%"
        ))
    
    print(separator)
    print("| {:<10} | {:>8} | {:>8} | {:>8} | {:>12} | {:>12} | {:>12} |".format("TOTAL", total_trades_all, total_wins_all, total_losses_all, f"{global_win_rate:.2f}%", "", ""))
    print(separator)
    print(f"| {'TOTAL EXPECTANCY (Wgt Avg)':<56} | {'{0:.4f}R'.format(float(weighted_expectancy)):>12} | {'':<12} |")
    print("="*105 + "\n")

def get_asset_type(pair: str) -> str:
    p_up = pair.upper()
    if any(k in p_up for k in ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'LTC', 'BCH', 'DOGE', 'DOT', 'LINK', 'MATIC', 'UNI', 'AVAX', 'TRX']): return "CRYPTO"
    if any(k in p_up for k in ['US30', 'SP500', 'SPX', 'NAS100', 'NSDQ', 'NDX', 'DOW', 'DAX', 'GER30', 'GER40', 'CAC', 'UK100', 'FTSE', 'JP225', 'NIKKEI', 'ASX', 'IBEX', 'STOXX']): return "INDICES"
    if any(k in p_up for k in ['XAU', 'XAG', 'WTI', 'BRENT', 'OIL', 'NATGAS', 'COPPER']): return "COMMODITIES"
    return "FOREX"

def display_keepers_csv(results: List[Dict[str, Any]]):
    keepers = [r for r in results if r['win_rate'] > 60.0]
    print("type,pair")
    for res in keepers:
        print(f"{get_asset_type(res['pair'])},{res['pair']}")

# ---------- MAIN ----------
def main():
    ap = argparse.ArgumentParser(description="Backtester FVG/EMA 50 - Mode Replace Pending Corrected")
    ap.add_argument("--pairs-file", default="pairs.txt", help="Fichier CSV listant les paires à tester.")
    ap.add_argument("--rr", type=Decimal, default=DEFAULT_RR, help=f"Ratio Risk/Reward (défaut: {DEFAULT_RR}).")
    ap.add_argument("--stdev-threshold", type=float, default=DEFAULT_STDEV_THRESHOLD, help=f"Seuil StDev.")
    ap.add_argument("--stdev-max", type=float, default=DEFAULT_STDEV_MAX, help=f"Seuil Max StDev.")
    ap.add_argument("--start-date", type=str, default=None, help="Date début.")
    ap.add_argument("--end-date", type=str, default=None, help="Date fin.")
    ap.add_argument("--risk", type=Decimal, default=DEFAULT_RISK_PER_TRADE, help=f"Risque.")

    args = ap.parse_args()
    
    start_ms = None
    if args.start_date:
        try: start_ms = parse_date_to_ms(args.start_date)
        except ValueError as e: print(f"[ERROR] Date début: {e}"); sys.exit(1)

    end_ms = None
    if args.end_date:
        try: end_ms = parse_date_to_ms(args.end_date, is_end_date=True) 
        except ValueError as e: print(f"[ERROR] Date fin: {e}"); sys.exit(1)
            
    if start_ms is not None and end_ms is not None and start_ms >= end_ms:
        print("[ERROR] Date début > Date fin.")
        sys.exit(1)

    engine = get_pg_engine()
    pairs = parse_pairs(args.pairs_file)
    if not pairs:
        print("No pairs found.")
        sys.exit(1)
        
    print(f"Lancement Backtest REPLACE-MODE (Corrected) sur {len(pairs)} paires.")
    
    # 1. Backtest
    for p in pairs:
        rates = fetch_all_rates(engine, p, SCAN_TF, start_ms, end_ms)
        if rates:
            execute_backtest(engine, rates, p, args.rr, 3, args.stdev_threshold, start_ms, end_ms, args.risk, args.stdev_max)

    # 2. Résultats
    display_summary_table(args.rr, args.stdev_threshold, args.risk, args.stdev_max, GLOBAL_RESULTS)
    display_keepers_csv(GLOBAL_RESULTS)

if __name__ == "__main__":
    main()