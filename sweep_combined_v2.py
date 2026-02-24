"""
Combined sweep v2: MR + CB with TP1_RR added, split fixed at 50/50.
Sweeps: SL_OFFSET x MIN_RR x TP1_RR for both strategies.
"""
import sys
import time
sys.path.insert(0, '.')

from optimize_combined_mr_cb import (
    load_data, precompute_candle_data, fast_backtest_combined
)

# Base params (fixed)
BASE = {
    'wait_candles': 3,
    'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
    'allowed_days': [0, 1, 2, 3, 4],
    'enable_mr': True,
    'mr_tp1_split': 0.5,   # Fixed 50/50
    'mr_use_trailing': True,
    'mr_min_poc_strength': 2.0,
    'mr_filter_entry_vs_poc': True,
    'mr_max_breakout_duration_min': 3,
    'mr_excluded_hours': [],
    'enable_cb': True,
    'cb_tp1_split': 0.5,   # Fixed 50/50 (was 0.3)
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
MR_TP1_RRS = [0.8, 1.0, 1.3, 1.5]

CB_SL_OFFSETS = [0.75, 1.0]
CB_MIN_RRS = [2.0, 2.5, 3.0]
CB_TP1_RRS = [0.8, 1.0, 1.3, 1.5]


def main():
    t0 = time.time()

    # Count combos
    mr_combos = len(MR_SL_OFFSETS) * len(MR_MIN_RRS) * len(MR_TP1_RRS)  # 36
    cb_combos = len(CB_SL_OFFSETS) * len(CB_MIN_RRS) * len(CB_TP1_RRS)   # 24
    # Too many if full cross product (36*24=864). Instead: sweep MR separately, sweep CB separately, then combine top.

    print("=" * 130)
    print("PARAMETER SWEEP V2: Split fixed 50/50, TP1_RR added")
    print("=" * 130)

    df_candles, ticks_by_minute, requested_start = load_data()
    print("[PRECOMPUTE] Computing candle data (VA=0.70)...")
    candle_data = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70)
    print(f"   {len(candle_data):,} candles precomputed in {time.time() - t0:.0f}s\n")

    # =========================================================================
    # Phase 1: Sweep MR (CB at best known: SL=0.75, RR=2.0, TP1=1.0)
    # =========================================================================
    print("=" * 140)
    print("PHASE 1: SWEEP MR (CB fixed: SL=0.75, RR=2.0, TP1=1.0, split 50/50)")
    print("=" * 140)
    header = (f"{'MR_SL':>5} | {'MR_RR':>5} | {'MR_TP1':>6} | "
              f"{'TOTAL':>5} | {'MR':>4} | {'CB':>4} | "
              f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
              f"{'MR_PnL':>7} | {'MR_PF':>5} | {'MR_WR':>5} | "
              f"{'CB_PnL':>7} | {'CB_PF':>5}")
    print(header)
    print("-" * len(header))

    mr_results = []
    for sl in MR_SL_OFFSETS:
        for rr in MR_MIN_RRS:
            for tp1 in MR_TP1_RRS:
                p = dict(BASE)
                p['mr_sl_offset'] = sl
                p['mr_min_rr'] = rr
                p['mr_tp1_rr'] = tp1
                # CB fixed at best
                p['cb_sl_offset'] = 0.75
                p['cb_min_rr'] = 2.0
                p['cb_tp1_rr'] = 1.0
                r = fast_backtest_combined(candle_data, p)
                mr_results.append((sl, rr, tp1, r))

                pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
                mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
                cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
                print(f"{sl:>5.2f} | {rr:>5.1f} | {tp1:>6.1f} | "
                      f"{r['total_trades']:>5} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
                      f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {pf:>5} | {r['max_dd_pct']:>4.2f}% | "
                      f"{r['mr_pnl_r']:>+6.1f}R | {mr_pf:>5} | {r['mr_wr']:>4.1f}% | "
                      f"{r['cb_pnl_r']:>+6.1f}R | {cb_pf:>5}")

    # Top 10 MR
    mr_results.sort(key=lambda x: x[3]['total_pnl_r'], reverse=True)
    print(f"\n--- TOP 10 MR configs (by total PnL R) ---")
    for i, (sl, rr, tp1, r) in enumerate(mr_results[:10]):
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
        print(f"  {i+1:>2}. MR(SL={sl}, RR={rr}, TP1={tp1}) | {r['total_trades']} trades ({r['mr_trades']}MR/{r['cb_trades']}CB) | "
              f"+{r['total_pnl_r']:.1f}R | PF {pf} | DD {r['max_dd_pct']:.2f}% | MR: +{r['mr_pnl_r']:.1f}R PF {mr_pf}")

    # =========================================================================
    # Phase 2: Sweep CB (MR at best from phase 1, top 3)
    # =========================================================================
    top3_mr = mr_results[:3]
    print(f"\n{'=' * 140}")
    print(f"PHASE 2: SWEEP CB (using top 3 MR configs from Phase 1)")
    print("=" * 140)

    combined_results = []
    for mr_rank, (mr_sl, mr_rr, mr_tp1, _) in enumerate(top3_mr):
        print(f"\n--- MR config #{mr_rank+1}: SL={mr_sl}, RR={mr_rr}, TP1={mr_tp1} ---")
        header = (f"{'CB_SL':>5} | {'CB_RR':>5} | {'CB_TP1':>6} | "
                  f"{'TOTAL':>5} | {'MR':>4} | {'CB':>4} | "
                  f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
                  f"{'MR_PnL':>7} | {'MR_PF':>5} | "
                  f"{'CB_PnL':>7} | {'CB_PF':>5}")
        print(header)
        print("-" * len(header))

        for cb_sl in CB_SL_OFFSETS:
            for cb_rr in CB_MIN_RRS:
                for cb_tp1 in CB_TP1_RRS:
                    p = dict(BASE)
                    p['mr_sl_offset'] = mr_sl
                    p['mr_min_rr'] = mr_rr
                    p['mr_tp1_rr'] = mr_tp1
                    p['cb_sl_offset'] = cb_sl
                    p['cb_min_rr'] = cb_rr
                    p['cb_tp1_rr'] = cb_tp1
                    r = fast_backtest_combined(candle_data, p)
                    combined_results.append((mr_sl, mr_rr, mr_tp1, cb_sl, cb_rr, cb_tp1, r))

                    pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
                    mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
                    cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
                    print(f"{cb_sl:>5.2f} | {cb_rr:>5.1f} | {cb_tp1:>6.1f} | "
                          f"{r['total_trades']:>5} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
                          f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {pf:>5} | {r['max_dd_pct']:>4.2f}% | "
                          f"{r['mr_pnl_r']:>+6.1f}R | {mr_pf:>5} | "
                          f"{r['cb_pnl_r']:>+6.1f}R | {cb_pf:>5}")

    # =========================================================================
    # Final ranking
    # =========================================================================
    print(f"\n{'=' * 160}")
    print("FINAL TOP 20 (sorted by PnL R)")
    print("=" * 160)
    header = (f"{'#':>2} | {'MR_SL':>5} {'MR_RR':>5} {'MR_TP1':>6} | {'CB_SL':>5} {'CB_RR':>5} {'CB_TP1':>6} | "
              f"{'TOTAL':>5} {'MR':>4} {'CB':>5} | "
              f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
              f"{'MR_PnL':>7} {'MR_PF':>5} | {'CB_PnL':>7} {'CB_PF':>5}")
    print(header)
    print("-" * len(header))

    combined_results.sort(key=lambda x: x[6]['total_pnl_r'], reverse=True)
    for i, (mr_sl, mr_rr, mr_tp1, cb_sl, cb_rr, cb_tp1, r) in enumerate(combined_results[:20]):
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
        cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
        print(f"{i+1:>2} | {mr_sl:>5.2f} {mr_rr:>5.1f} {mr_tp1:>6.1f} | {cb_sl:>5.2f} {cb_rr:>5.1f} {cb_tp1:>6.1f} | "
              f"{r['total_trades']:>5} {r['mr_trades']:>4} {r['cb_trades']:>5} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {pf:>5} | {r['max_dd_pct']:>4.2f}% | "
              f"{r['mr_pnl_r']:>+6.1f}R {mr_pf:>5} | {r['cb_pnl_r']:>+6.1f}R {cb_pf:>5}")

    # Composite ranking
    def score(x):
        r = x[6]
        pf = min(r['profit_factor'], 5.0) if r['profit_factor'] != float('inf') else 5.0
        return r['total_pnl_r'] * pf * max(10.0 - r['max_dd_pct'], 0.1)

    print(f"\n{'=' * 160}")
    print("FINAL TOP 20 (by composite: PnL_R * PF * (10 - MaxDD%))")
    print("=" * 160)
    print(header)
    print("-" * len(header))

    combined_results.sort(key=score, reverse=True)
    for i, (mr_sl, mr_rr, mr_tp1, cb_sl, cb_rr, cb_tp1, r) in enumerate(combined_results[:20]):
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
        cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
        s = score((mr_sl, mr_rr, mr_tp1, cb_sl, cb_rr, cb_tp1, r))
        print(f"{i+1:>2} | {mr_sl:>5.2f} {mr_rr:>5.1f} {mr_tp1:>6.1f} | {cb_sl:>5.2f} {cb_rr:>5.1f} {cb_tp1:>6.1f} | "
              f"{r['total_trades']:>5} {r['mr_trades']:>4} {r['cb_trades']:>5} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {pf:>5} | {r['max_dd_pct']:>4.2f}% | "
              f"{r['mr_pnl_r']:>+6.1f}R {mr_pf:>5} | {r['cb_pnl_r']:>+6.1f}R {cb_pf:>5}  (score: {s:.0f})")

    print(f"\nTotal time: {time.time() - t0:.0f}s")
    print(f"NOTE: All configs use 50/50 split for both MR and CB")


if __name__ == '__main__':
    main()
