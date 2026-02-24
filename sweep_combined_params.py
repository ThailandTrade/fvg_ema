"""
Combined sweep: MR(SL_OFF x MIN_RR) x CB(SL_OFF x MIN_RR) tested together.
Since strategies interact (single active trade), we must test combos together.
"""
import sys
import time
sys.path.insert(0, '.')

from optimize_combined_mr_cb import (
    load_data, precompute_candle_data, fast_backtest_combined
)

# Base params (everything except what we sweep)
BASE = {
    'wait_candles': 3,
    'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
    'allowed_days': [0, 1, 2, 3, 4],
    'enable_mr': True,
    'mr_tp1_rr': 1.3,
    'mr_tp1_split': 0.5,
    'mr_use_trailing': True,
    'mr_min_poc_strength': 2.0,
    'mr_filter_entry_vs_poc': True,
    'mr_max_breakout_duration_min': 3,
    'mr_excluded_hours': [],
    'enable_cb': True,
    'cb_tp1_rr': 1.0,
    'cb_tp1_split': 0.3,
    'cb_use_trailing': True,
    'cb_min_poc_strength': 3.0,
    'cb_excluded_hours': [0, 10],
    'cb_exclude_vah_target': True,
    'cb_use_prev_day': True,
    'cb_use_prev_week': True,
}

# Sweep values
MR_SL_OFFSETS = [0.50, 0.75, 1.0]
MR_MIN_RRS = [1.5, 2.0, 2.5]
CB_SL_OFFSETS = [0.75, 1.0]
CB_MIN_RRS = [2.0, 2.5, 3.0]


def main():
    t0 = time.time()
    print("=" * 130)
    print("COMBINED SWEEP: MR(SL_OFF x MIN_RR) x CB(SL_OFF x MIN_RR)")
    print(f"  MR SL_OFF: {MR_SL_OFFSETS} x MR MIN_RR: {MR_MIN_RRS}")
    print(f"  CB SL_OFF: {CB_SL_OFFSETS} x CB MIN_RR: {CB_MIN_RRS}")
    total_combos = len(MR_SL_OFFSETS) * len(MR_MIN_RRS) * len(CB_SL_OFFSETS) * len(CB_MIN_RRS)
    print(f"  Total combos: {total_combos}")
    print("=" * 130)

    df_candles, ticks_by_minute, requested_start = load_data()
    print("[PRECOMPUTE] Computing candle data (VA=0.70)...")
    candle_data = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70)
    print(f"   {len(candle_data):,} candles precomputed")
    print(f"   Precompute done in {time.time() - t0:.0f}s\n")

    results = []
    sweep_start = time.time()

    for mr_sl in MR_SL_OFFSETS:
        for mr_rr in MR_MIN_RRS:
            for cb_sl in CB_SL_OFFSETS:
                for cb_rr in CB_MIN_RRS:
                    p = dict(BASE)
                    p['mr_sl_offset'] = mr_sl
                    p['mr_min_rr'] = mr_rr
                    p['cb_sl_offset'] = cb_sl
                    p['cb_min_rr'] = cb_rr
                    r = fast_backtest_combined(candle_data, p)
                    results.append((mr_sl, mr_rr, cb_sl, cb_rr, r))

    sweep_time = time.time() - sweep_start
    print(f"Sweep done in {sweep_time:.0f}s ({total_combos} combos)\n")

    # Full table
    print("=" * 160)
    print("ALL RESULTS")
    print("=" * 160)
    header = (f"{'MR_SL':>5} | {'MR_RR':>5} | {'CB_SL':>5} | {'CB_RR':>5} | "
              f"{'TOTAL':>5} | {'MR':>4} | {'CB':>4} | "
              f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
              f"{'MR_PnL':>7} | {'MR_PF':>5} | {'MR_WR':>5} | "
              f"{'CB_PnL':>7} | {'CB_PF':>5} | {'CB_WR':>5}")
    print(header)
    print("-" * len(header))

    # Sort by PnL R desc
    results.sort(key=lambda x: x[4]['total_pnl_r'], reverse=True)

    baseline_marker = (1.0, 2.5, 1.0, 2.0)
    for mr_sl, mr_rr, cb_sl, cb_rr, r in results:
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
        cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
        is_base = (mr_sl, mr_rr, cb_sl, cb_rr) == baseline_marker
        marker = " <<< BASELINE" if is_base else ""
        print(f"{mr_sl:>5.2f} | {mr_rr:>5.1f} | {cb_sl:>5.2f} | {cb_rr:>5.1f} | "
              f"{r['total_trades']:>5} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {pf:>5} | {r['max_dd_pct']:>4.2f}% | "
              f"{r['mr_pnl_r']:>+6.1f}R | {mr_pf:>5} | {r['mr_wr']:>4.1f}% | "
              f"{r['cb_pnl_r']:>+6.1f}R | {cb_pf:>5} | {r['cb_wr']:>4.1f}%{marker}")

    # Composite ranking
    print(f"\n{'=' * 130}")
    print("TOP 15 BY COMPOSITE SCORE (PnL_R * PF * (10 - MaxDD%))")
    print("=" * 130)
    header2 = (f"{'#':>2} | {'MR_SL':>5} | {'MR_RR':>5} | {'CB_SL':>5} | {'CB_RR':>5} | "
               f"{'TOTAL':>5} | {'MR':>4} | {'CB':>4} | "
               f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | {'SCORE':>6}")
    print(header2)
    print("-" * len(header2))

    def score(x):
        r = x[4]
        pf = min(r['profit_factor'], 5.0) if r['profit_factor'] != float('inf') else 5.0
        return r['total_pnl_r'] * pf * max(10.0 - r['max_dd_pct'], 0.1)

    results.sort(key=score, reverse=True)
    for i, (mr_sl, mr_rr, cb_sl, cb_rr, r) in enumerate(results[:15]):
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        s = score((mr_sl, mr_rr, cb_sl, cb_rr, r))
        is_base = (mr_sl, mr_rr, cb_sl, cb_rr) == baseline_marker
        marker = " <<< BASELINE" if is_base else ""
        print(f"{i+1:>2} | {mr_sl:>5.2f} | {mr_rr:>5.1f} | {cb_sl:>5.2f} | {cb_rr:>5.1f} | "
              f"{r['total_trades']:>5} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {pf:>5} | {r['max_dd_pct']:>4.2f}% | {s:>6.0f}{marker}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
