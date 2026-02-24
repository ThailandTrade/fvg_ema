"""
Validate top configs from sweep v3 on last 3 months (Dec 2025 - Feb 2026).
"""
import sys
import time
sys.path.insert(0, '.')

from optimize_combined_mr_cb import (
    load_data, precompute_candle_data, fast_backtest_combined, run_on_period
)

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

# Top configs from sweep v3
CONFIGS = [
    {
        'name': '#1 MR(1.0/1.5/1.5/70-30) CB(0.75/2.0/1.0/30-70)',
        'mr_sl_offset': 1.0, 'mr_min_rr': 1.5, 'mr_tp1_rr': 1.5, 'mr_tp1_split': 0.7,
        'cb_sl_offset': 0.75, 'cb_min_rr': 2.0, 'cb_tp1_rr': 1.0, 'cb_tp1_split': 0.3,
    },
    {
        'name': '#2 MR(1.0/1.5/1.5/50-50) CB(0.75/2.0/1.0/30-70)',
        'mr_sl_offset': 1.0, 'mr_min_rr': 1.5, 'mr_tp1_rr': 1.5, 'mr_tp1_split': 0.5,
        'cb_sl_offset': 0.75, 'cb_min_rr': 2.0, 'cb_tp1_rr': 1.0, 'cb_tp1_split': 0.3,
    },
    {
        'name': '#4 MR(1.0/1.5/1.5/30-70) CB(0.75/2.0/1.3/30-70)',
        'mr_sl_offset': 1.0, 'mr_min_rr': 1.5, 'mr_tp1_rr': 1.5, 'mr_tp1_split': 0.3,
        'cb_sl_offset': 0.75, 'cb_min_rr': 2.0, 'cb_tp1_rr': 1.3, 'cb_tp1_split': 0.3,
    },
    {
        'name': '#7 MR(1.0/1.5/1.5/70-30) CB(0.75/3.0/1.0/30-70)',
        'mr_sl_offset': 1.0, 'mr_min_rr': 1.5, 'mr_tp1_rr': 1.5, 'mr_tp1_split': 0.7,
        'cb_sl_offset': 0.75, 'cb_min_rr': 3.0, 'cb_tp1_rr': 1.0, 'cb_tp1_split': 0.3,
    },
    {
        'name': '#8 MR(1.0/1.5/1.5/70-30) CB(1.0/2.0/1.0/30-70)',
        'mr_sl_offset': 1.0, 'mr_min_rr': 1.5, 'mr_tp1_rr': 1.5, 'mr_tp1_split': 0.7,
        'cb_sl_offset': 1.0, 'cb_min_rr': 2.0, 'cb_tp1_rr': 1.0, 'cb_tp1_split': 0.3,
    },
    {
        'name': 'BASELINE MR(1.0/2.5/1.3/50-50) CB(1.0/2.0/1.0/30-70)',
        'mr_sl_offset': 1.0, 'mr_min_rr': 2.5, 'mr_tp1_rr': 1.3, 'mr_tp1_split': 0.5,
        'cb_sl_offset': 1.0, 'cb_min_rr': 2.0, 'cb_tp1_rr': 1.0, 'cb_tp1_split': 0.3,
    },
]

PERIODS = [
    ('FULL (13 mois)', '2025-01-01', '2026-02-28'),
    ('LAST 3 MOIS', '2025-12-01', '2026-02-28'),
    ('Dec 2025', '2025-12-01', '2025-12-31'),
    ('Jan 2026', '2026-01-01', '2026-01-31'),
    ('Feb 2026', '2026-02-01', '2026-02-28'),
]


def fmt_pf(v):
    if v is None: return "N/A"
    return f"{v:.2f}" if v != float('inf') else "inf"


def main():
    t0 = time.time()
    print("=" * 140)
    print("VALIDATION: Top configs on recent periods")
    print("=" * 140)

    df_candles, ticks_by_minute, requested_start = load_data()
    print("[PRECOMPUTE]...")
    candle_data = precompute_candle_data(df_candles, ticks_by_minute, requested_start, va_percent=0.70)
    print(f"   {len(candle_data):,} candles in {time.time() - t0:.0f}s\n")

    for cfg in CONFIGS:
        name = cfg.pop('name')
        p = dict(BASE)
        p.update(cfg)
        cfg['name'] = name  # restore

        print(f"\n{'=' * 140}")
        print(f"CONFIG: {name}")
        print(f"{'=' * 140}")
        header = (f"{'PERIOD':<20} | {'TRADES':>6} | {'MR':>4} | {'CB':>4} | "
                  f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5} | "
                  f"{'MR_PnL':>7} {'MR_PF':>5} {'MR_WR':>5} | "
                  f"{'CB_PnL':>7} {'CB_PF':>5} {'CB_WR':>5}")
        print(header)
        print("-" * len(header))

        for period_name, start, end in PERIODS:
            r = run_on_period(candle_data, start, end, p)
            if r is None or r['total_trades'] == 0:
                print(f"{period_name:<20} | {'NO DATA':>6}")
                continue
            print(f"{period_name:<20} | {r['total_trades']:>6} | {r['mr_trades']:>4} | {r['cb_trades']:>4} | "
                  f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {fmt_pf(r['profit_factor']):>5} | {r['max_dd_pct']:>4.2f}% | "
                  f"{r['mr_pnl_r']:>+6.1f}R {fmt_pf(r['mr_pf']):>5} {r['mr_wr']:>4.1f}% | "
                  f"{r['cb_pnl_r']:>+6.1f}R {fmt_pf(r['cb_pf']):>5} {r['cb_wr']:>4.1f}%")

    # Summary comparison on LAST 3 MONTHS
    print(f"\n{'=' * 140}")
    print("SUMMARY: LAST 3 MONTHS comparison")
    print("=" * 140)
    header = (f"{'CONFIG':<55} | {'TRADES':>6} | {'MR':>4} {'CB':>4} | "
              f"{'WR%':>5} | {'PnL_R':>7} | {'PF':>5} | {'DD%':>5}")
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        name = cfg.pop('name')
        p = dict(BASE)
        p.update(cfg)
        cfg['name'] = name

        r = run_on_period(candle_data, '2025-12-01', '2026-02-28', p)
        if r is None:
            print(f"{name:<55} | NO DATA")
            continue
        print(f"{name:<55} | {r['total_trades']:>6} | {r['mr_trades']:>4} {r['cb_trades']:>4} | "
              f"{r['win_rate']:>4.1f}% | {r['total_pnl_r']:>+6.1f}R | {fmt_pf(r['profit_factor']):>5} | {r['max_dd_pct']:>4.2f}%")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
