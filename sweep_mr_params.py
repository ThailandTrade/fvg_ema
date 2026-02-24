"""
Quick parameter sweep for MR SL_OFFSET x MIN_RR combinations.
Uses optimizer's precompute + fast_backtest to avoid reloading data.
Also sweeps CB SL_OFFSET x MIN_RR for comparison.
"""
import sys
import time
sys.path.insert(0, '.')

from optimize_combined_mr_cb import (
    load_data, precompute_candle_data, fast_backtest_combined
)

# Current optimized baseline
BASELINE = {
    'wait_candles': 3,
    'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
    'allowed_days': [0, 1, 2, 3, 4],
    # MR
    'enable_mr': True,
    'mr_min_rr': 2.5,
    'mr_sl_offset': 1.0,
    'mr_tp1_rr': 1.3,
    'mr_tp1_split': 0.5,
    'mr_use_trailing': True,
    'mr_min_poc_strength': 2.0,
    'mr_filter_entry_vs_poc': True,
    'mr_max_breakout_duration_min': 3,
    'mr_excluded_hours': [],
    # CB
    'enable_cb': True,
    'cb_min_rr': 2.0,
    'cb_sl_offset': 1.0,
    'cb_tp1_rr': 1.0,
    'cb_tp1_split': 0.3,
    'cb_use_trailing': True,
    'cb_min_poc_strength': 3.0,
    'cb_excluded_hours': [0, 10],
    'cb_exclude_vah_target': True,
    'cb_use_prev_day': True,
    'cb_use_prev_week': True,
}

MR_SL_OFFSETS = [0.25, 0.50, 0.75, 1.0, 1.5]
MR_MIN_RRS = [1.5, 2.0, 2.5, 3.0]
CB_SL_OFFSETS = [0.25, 0.50, 0.75, 1.0, 1.5]
CB_MIN_RRS = [1.5, 2.0, 2.5, 3.0]


def main():
    t0 = time.time()
    print("=" * 120)
    print("PARAMETER SWEEP: MR & CB SL_OFFSET x MIN_RR")
    print("=" * 120)

    # Load + precompute once
    df_candles, ticks_by_minute, requested_start = load_data()
    print("[PRECOMPUTE] Computing candle data (VA=0.70)...")
    candle_data = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70)
    print(f"   {len(candle_data):,} candles precomputed")
    precompute_time = time.time() - t0
    print(f"   Precompute done in {precompute_time:.0f}s\n")

    # =========================================================================
    # SWEEP MR params (CB fixed at baseline)
    # =========================================================================
    print("=" * 120)
    print("SWEEP MR: SL_OFFSET x MIN_RR  (CB params fixed)")
    print("=" * 120)
    header = f"{'SL_OFF':>6} | {'MIN_RR':>6} | {'TRADES':>6} | {'MR_TR':>5} | {'CB_TR':>5} | {'WR%':>6} | {'PnL_R':>8} | {'PF':>6} | {'MaxDD%':>6} | {'MR_PnL':>8} | {'MR_PF':>6} | {'MR_WR%':>6} | {'CB_PnL':>8} | {'CB_PF':>6}"
    print(header)
    print("-" * len(header))

    mr_results = []
    for sl_off in MR_SL_OFFSETS:
        for min_rr in MR_MIN_RRS:
            p = dict(BASELINE)
            p['mr_sl_offset'] = sl_off
            p['mr_min_rr'] = min_rr
            r = fast_backtest_combined(candle_data, p)
            mr_results.append((sl_off, min_rr, r))

            mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
            cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
            pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"

            is_baseline = sl_off == 1.0 and min_rr == 2.5
            marker = " <<< BASELINE" if is_baseline else ""

            print(f"{sl_off:>6.2f} | {min_rr:>6.1f} | {r['total_trades']:>6} | {r['mr_trades']:>5} | {r['cb_trades']:>5} | "
                  f"{r['win_rate']:>5.1f}% | {r['total_pnl_r']:>+7.1f}R | {pf:>6} | {r['max_dd_pct']:>5.2f}% | "
                  f"{r['mr_pnl_r']:>+7.1f}R | {mr_pf:>6} | {r['mr_wr']:>5.1f}% | "
                  f"{r['cb_pnl_r']:>+7.1f}R | {cb_pf:>6}{marker}")

    # =========================================================================
    # SWEEP CB params (MR fixed at baseline)
    # =========================================================================
    print(f"\n{'=' * 120}")
    print("SWEEP CB: SL_OFFSET x MIN_RR  (MR params fixed at baseline)")
    print("=" * 120)
    header = f"{'SL_OFF':>6} | {'MIN_RR':>6} | {'TRADES':>6} | {'MR_TR':>5} | {'CB_TR':>5} | {'WR%':>6} | {'PnL_R':>8} | {'PF':>6} | {'MaxDD%':>6} | {'CB_PnL':>8} | {'CB_PF':>6} | {'CB_WR%':>6} | {'MR_PnL':>8} | {'MR_PF':>6}"
    print(header)
    print("-" * len(header))

    cb_results = []
    for sl_off in CB_SL_OFFSETS:
        for min_rr in CB_MIN_RRS:
            p = dict(BASELINE)
            p['cb_sl_offset'] = sl_off
            p['cb_min_rr'] = min_rr
            r = fast_backtest_combined(candle_data, p)
            cb_results.append((sl_off, min_rr, r))

            mr_pf = f"{r['mr_pf']:.2f}" if r['mr_pf'] != float('inf') else "inf"
            cb_pf = f"{r['cb_pf']:.2f}" if r['cb_pf'] != float('inf') else "inf"
            pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"

            is_baseline = sl_off == 1.0 and min_rr == 2.0
            marker = " <<< BASELINE" if is_baseline else ""

            print(f"{sl_off:>6.2f} | {min_rr:>6.1f} | {r['total_trades']:>6} | {r['mr_trades']:>5} | {r['cb_trades']:>5} | "
                  f"{r['win_rate']:>5.1f}% | {r['total_pnl_r']:>+7.1f}R | {pf:>6} | {r['max_dd_pct']:>5.2f}% | "
                  f"{r['cb_pnl_r']:>+7.1f}R | {cb_pf:>6} | {r['cb_wr']:>5.1f}% | "
                  f"{r['mr_pnl_r']:>+7.1f}R | {mr_pf:>6}{marker}")

    # =========================================================================
    # TOP 10 overall (MR sweep + CB sweep combined, sorted by PnL_R)
    # =========================================================================
    print(f"\n{'=' * 120}")
    print("TOP 10 BEST CONFIGS (by PnL R)")
    print("=" * 120)

    all_results = []
    for sl_off, min_rr, r in mr_results:
        all_results.append(('MR', sl_off, min_rr, r))
    for sl_off, min_rr, r in cb_results:
        all_results.append(('CB', sl_off, min_rr, r))

    # Sort by total_pnl_r descending
    all_results.sort(key=lambda x: x[3]['total_pnl_r'], reverse=True)

    header = f"{'TYPE':>4} | {'SL_OFF':>6} | {'MIN_RR':>6} | {'TRADES':>6} | {'MR/CB':>10} | {'WR%':>6} | {'PnL_R':>8} | {'PF':>6} | {'MaxDD%':>6} | {'MR_PnL':>8} | {'CB_PnL':>8}"
    print(header)
    print("-" * len(header))

    for typ, sl_off, min_rr, r in all_results[:10]:
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        print(f"{typ:>4} | {sl_off:>6.2f} | {min_rr:>6.1f} | {r['total_trades']:>6} | "
              f"{r['mr_trades']:>4}/{r['cb_trades']:<5} | {r['win_rate']:>5.1f}% | "
              f"{r['total_pnl_r']:>+7.1f}R | {pf:>6} | {r['max_dd_pct']:>5.2f}% | "
              f"{r['mr_pnl_r']:>+7.1f}R | {r['cb_pnl_r']:>+7.1f}R")

    # =========================================================================
    # TOP 10 by composite (PnL * PF * (10-DD))
    # =========================================================================
    print(f"\n{'=' * 120}")
    print("TOP 10 BEST CONFIGS (by composite: PnL_R * PF * (10 - MaxDD%))")
    print("=" * 120)
    print(header)
    print("-" * len(header))

    def score(x):
        r = x[3]
        pf = min(r['profit_factor'], 5.0) if r['profit_factor'] != float('inf') else 5.0
        return r['total_pnl_r'] * pf * max(10.0 - r['max_dd_pct'], 0.1)

    all_results.sort(key=score, reverse=True)
    for typ, sl_off, min_rr, r in all_results[:10]:
        pf = f"{r['profit_factor']:.2f}" if r['profit_factor'] != float('inf') else "inf"
        s = score(('', '', '', r))
        print(f"{typ:>4} | {sl_off:>6.2f} | {min_rr:>6.1f} | {r['total_trades']:>6} | "
              f"{r['mr_trades']:>4}/{r['cb_trades']:<5} | {r['win_rate']:>5.1f}% | "
              f"{r['total_pnl_r']:>+7.1f}R | {pf:>6} | {r['max_dd_pct']:>5.2f}% | "
              f"{r['mr_pnl_r']:>+7.1f}R | {r['cb_pnl_r']:>+7.1f}R  (score: {s:.0f})")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
