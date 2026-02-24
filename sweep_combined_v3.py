"""
Combined sweep v3: Full sweep including TP1_SPLIT for both strategies.
Phase 1: Sweep MR (SL x RR x TP1_RR x SPLIT) with CB fixed
Phase 2: Take top 3 MR, sweep CB (SL x RR x TP1_RR x SPLIT)
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
    'mr_use_trailing': True,
    'mr_min_poc_strength': 2.0,
    'mr_filter_entry_vs_poc': True,
    'mr_max_breakout_duration_min': 3,
    'mr_excluded_hours': [],
    'enable_cb': True,
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
MR_SPLITS = [0.3, 0.5, 0.7]

CB_SL_OFFSETS = [0.75, 1.0]
CB_MIN_RRS = [2.0, 2.5, 3.0]
CB_TP1_RRS = [0.8, 1.0, 1.3, 1.5]
CB_SPLITS = [0.3, 0.5, 0.7]


def fmt_pf(v):
    return f"{v:.2f}" if v != float('inf') else "inf"


def main():
    t0 = time.time()

    mr_combos = len(MR_SL_OFFSETS) * len(MR_MIN_RRS) * len(MR_TP1_RRS) * len(MR_SPLITS)
    cb_combos = len(CB_SL_OFFSETS) * len(CB_MIN_RRS) * len(CB_TP1_RRS) * len(CB_SPLITS)

    print("=" * 130)
    print("PARAMETER SWEEP V3: Full sweep with TP1_SPLIT")
    print(f"  Phase 1: {mr_combos} MR combos (CB fixed)")
    print(f"  Phase 2: {cb_combos} CB combos x top 3 MR = {cb_combos * 3}")
    print(f"  Total: {mr_combos + cb_combos * 3} combos")
    print("=" * 130)

    df_candles, ticks_by_minute, requested_start = load_data()
    print("[PRECOMPUTE] Computing candle data (VA=0.70)...")
    candle_data = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70)
    print(f"   {len(candle_data):,} candles precomputed in {time.time() - t0:.0f}s\n")

    # =========================================================================
    # Phase 1: Sweep MR (CB at reasonable default: SL=0.75, RR=2.0, TP1=1.0, split=0.3)
    # =========================================================================
    print("=" * 150)
    print(f"PHASE 1: SWEEP MR ({mr_combos} combos) | CB fixed: SL=0.75, RR=2.0, TP1=1.0, split=30/70")
    print("=" * 150)
    header = (f"{'MR_SL':>5} | {'MR_RR':>5} | {'TP1_RR':>6} | {'SPLIT':>5} | "
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
                for split in MR_SPLITS:
                    p = dict(BASE)
                    p['mr_sl_offset'] = sl
                    p['mr_min_rr'] = rr
                    p['mr_tp1_rr'] = tp1
                    p['mr_tp1_split'] = split
                    # CB fixed
                    p['cb_sl_offset'] = 0.75
                    p['cb_min_rr'] = 2.0
                    p['cb_tp1_rr'] = 1.0
                    p['cb_tp1_split'] = 0.3
                    r = fast_backtest_combined(candle_data, p)
                    mr_results.append((sl, rr, tp1, split, r))

                    tp2 = 1.0 - split
                    print(f"{sl:>5.2f} | {rr:>5.1f} | {tp1:>6.1f} | {int(split*100):>2}/{int(tp2*100):<2} | "
                          f"{r['total_trades']:>5} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
                          f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {fmt_pf(r['profit_factor']):>5} | {r['max_dd_pct']:>4.2f}% | "
                          f"{r['mr_pnl_r']:>+6.1f}R | {fmt_pf(r['mr_pf']):>5} | {r['mr_wr']:>4.1f}% | "
                          f"{r['cb_pnl_r']:>+6.1f}R | {fmt_pf(r['cb_pf']):>5}")

    # Top 15 MR
    mr_results.sort(key=lambda x: x[4]['total_pnl_r'], reverse=True)
    print(f"\n--- TOP 15 MR configs (by total PnL R) ---")
    for i, (sl, rr, tp1, split, r) in enumerate(mr_results[:15]):
        tp2 = 1.0 - split
        print(f"  {i+1:>2}. MR(SL={sl}, RR={rr}, TP1={tp1}, split={int(split*100)}/{int(tp2*100)}) | "
              f"{r['total_trades']} tr ({r['mr_trades']}MR/{r['cb_trades']}CB) | "
              f"+{r['total_pnl_r']:.1f}R | PF {fmt_pf(r['profit_factor'])} | DD {r['max_dd_pct']:.2f}% | "
              f"MR: +{r['mr_pnl_r']:.1f}R PF {fmt_pf(r['mr_pf'])} WR {r['mr_wr']:.1f}%")

    # =========================================================================
    # Phase 2: Sweep CB with top 3 MR configs
    # =========================================================================
    top3_mr = mr_results[:3]
    print(f"\n{'=' * 150}")
    print(f"PHASE 2: SWEEP CB ({cb_combos} combos x 3 MR configs = {cb_combos * 3})")
    print("=" * 150)

    combined_results = []
    for mr_rank, (mr_sl, mr_rr, mr_tp1, mr_split, _) in enumerate(top3_mr):
        mr_tp2 = 1.0 - mr_split
        print(f"\n--- MR #{mr_rank+1}: SL={mr_sl}, RR={mr_rr}, TP1={mr_tp1}, split={int(mr_split*100)}/{int(mr_tp2*100)} ---")
        header = (f"{'CB_SL':>5} | {'CB_RR':>5} | {'TP1_RR':>6} | {'SPLIT':>5} | "
                  f"{'TOTAL':>5} | {'MR':>4} | {'CB':>4} | "
                  f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
                  f"{'MR_PnL':>7} | {'MR_PF':>5} | "
                  f"{'CB_PnL':>7} | {'CB_PF':>5} | {'CB_WR':>5}")
        print(header)
        print("-" * len(header))

        for cb_sl in CB_SL_OFFSETS:
            for cb_rr in CB_MIN_RRS:
                for cb_tp1 in CB_TP1_RRS:
                    for cb_split in CB_SPLITS:
                        p = dict(BASE)
                        p['mr_sl_offset'] = mr_sl
                        p['mr_min_rr'] = mr_rr
                        p['mr_tp1_rr'] = mr_tp1
                        p['mr_tp1_split'] = mr_split
                        p['cb_sl_offset'] = cb_sl
                        p['cb_min_rr'] = cb_rr
                        p['cb_tp1_rr'] = cb_tp1
                        p['cb_tp1_split'] = cb_split
                        r = fast_backtest_combined(candle_data, p)
                        combined_results.append((mr_sl, mr_rr, mr_tp1, mr_split, cb_sl, cb_rr, cb_tp1, cb_split, r))

                        cb_tp2 = 1.0 - cb_split
                        print(f"{cb_sl:>5.2f} | {cb_rr:>5.1f} | {cb_tp1:>6.1f} | {int(cb_split*100):>2}/{int(cb_tp2*100):<2} | "
                              f"{r['total_trades']:>5} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
                              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {fmt_pf(r['profit_factor']):>5} | {r['max_dd_pct']:>4.2f}% | "
                              f"{r['mr_pnl_r']:>+6.1f}R | {fmt_pf(r['mr_pf']):>5} | "
                              f"{r['cb_pnl_r']:>+6.1f}R | {fmt_pf(r['cb_pf']):>5} | {r['cb_wr']:>4.1f}%")

    # =========================================================================
    # Final rankings
    # =========================================================================
    print(f"\n{'=' * 170}")
    print("FINAL TOP 20 (by PnL R)")
    print("=" * 170)
    header = (f"{'#':>2} | {'MR_SL':>5} {'MR_RR':>5} {'MR_TP1':>6} {'MR_SP':>5} | "
              f"{'CB_SL':>5} {'CB_RR':>5} {'CB_TP1':>6} {'CB_SP':>5} | "
              f"{'TOTAL':>5} {'MR':>4} {'CB':>5} | "
              f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
              f"{'MR_PnL':>7} {'MR_PF':>5} | {'CB_PnL':>7} {'CB_PF':>5}")
    print(header)
    print("-" * len(header))

    combined_results.sort(key=lambda x: x[8]['total_pnl_r'], reverse=True)
    for i, (mr_sl, mr_rr, mr_tp1, mr_sp, cb_sl, cb_rr, cb_tp1, cb_sp, r) in enumerate(combined_results[:20]):
        mr_sp2 = 1.0 - mr_sp
        cb_sp2 = 1.0 - cb_sp
        print(f"{i+1:>2} | {mr_sl:>5.2f} {mr_rr:>5.1f} {mr_tp1:>6.1f} {int(mr_sp*100):>2}/{int(mr_sp2*100):<2} | "
              f"{cb_sl:>5.2f} {cb_rr:>5.1f} {cb_tp1:>6.1f} {int(cb_sp*100):>2}/{int(cb_sp2*100):<2} | "
              f"{r['total_trades']:>5} {r['mr_trades']:>4} {r['cb_trades']:>5} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {fmt_pf(r['profit_factor']):>5} | {r['max_dd_pct']:>4.2f}% | "
              f"{r['mr_pnl_r']:>+6.1f}R {fmt_pf(r['mr_pf']):>5} | {r['cb_pnl_r']:>+6.1f}R {fmt_pf(r['cb_pf']):>5}")

    # Composite
    def score(x):
        r = x[8]
        pf = min(r['profit_factor'], 5.0) if r['profit_factor'] != float('inf') else 5.0
        return r['total_pnl_r'] * pf * max(10.0 - r['max_dd_pct'], 0.1)

    print(f"\n{'=' * 170}")
    print("FINAL TOP 20 (by composite: PnL_R * PF * (10 - MaxDD%))")
    print("=" * 170)
    print(header)
    print("-" * len(header))

    combined_results.sort(key=score, reverse=True)
    for i, (mr_sl, mr_rr, mr_tp1, mr_sp, cb_sl, cb_rr, cb_tp1, cb_sp, r) in enumerate(combined_results[:20]):
        mr_sp2 = 1.0 - mr_sp
        cb_sp2 = 1.0 - cb_sp
        s = score((mr_sl, mr_rr, mr_tp1, mr_sp, cb_sl, cb_rr, cb_tp1, cb_sp, r))
        print(f"{i+1:>2} | {mr_sl:>5.2f} {mr_rr:>5.1f} {mr_tp1:>6.1f} {int(mr_sp*100):>2}/{int(mr_sp2*100):<2} | "
              f"{cb_sl:>5.2f} {cb_rr:>5.1f} {cb_tp1:>6.1f} {int(cb_sp*100):>2}/{int(cb_sp2*100):<2} | "
              f"{r['total_trades']:>5} {r['mr_trades']:>4} {r['cb_trades']:>5} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {fmt_pf(r['profit_factor']):>5} | {r['max_dd_pct']:>4.2f}% | "
              f"{r['mr_pnl_r']:>+6.1f}R {fmt_pf(r['mr_pf']):>5} | {r['cb_pnl_r']:>+6.1f}R {fmt_pf(r['cb_pf']):>5}  (score: {s:.0f})")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
