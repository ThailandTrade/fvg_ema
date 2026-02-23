"""
Temporary script: run CB original and combined-CB-only with aligned configs,
then compare results.
"""
import subprocess, sys, re, os

os.environ['PYTHONIOENCODING'] = 'utf-8'
ENV = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}

def extract_metrics(output):
    """Extract key metrics from backtest output."""
    metrics = {}
    for line in output.split('\n'):
        if 'Trades Total:' in line:
            m = re.search(r'Trades Total:\s+(\d+)', line)
            if m: metrics['trades'] = int(m.group(1))
        elif 'Win Rate:' in line:
            m = re.search(r'Win Rate:\s+([\d.]+)%', line)
            if m: metrics['wr'] = float(m.group(1))
        elif 'Gain Total:' in line:
            m = re.search(r'Gain Total:\s+([+\-\d.]+)\s*R', line)
            if m: metrics['pnl_r'] = float(m.group(1))
        elif 'Profit Factor:' in line:
            m = re.search(r'Profit Factor:\s+([\d.inf]+)', line)
            if m:
                v = m.group(1)
                metrics['pf'] = float('inf') if 'inf' in v else float(v)
        elif 'Capital Final:' in line:
            m = re.search(r'Capital Final:\s+\$([\d,.\-]+)', line)
            if m: metrics['capital'] = float(m.group(1).replace(',', ''))
        elif 'Wins / BE / Losses:' in line:
            m = re.search(r'(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', line)
            if m:
                metrics['wins'] = int(m.group(1))
                metrics['be'] = int(m.group(2))
                metrics['losses'] = int(m.group(3))
        elif 'Wins / Losses:' in line and 'BE' not in line:
            m = re.search(r'(\d+)\s*/\s*(\d+)', line)
            if m:
                metrics['wins'] = int(m.group(1))
                metrics['be'] = 0
                metrics['losses'] = int(m.group(2))
        elif 'Max Drawdown:' in line:
            m = re.search(r'Max Drawdown:\s+([\d.]+)%', line)
            if m: metrics['dd'] = float(m.group(1))
        elif 'Expectancy:' in line:
            m = re.search(r'Expectancy:\s+([+\-\d.]+)\s*R', line)
            if m: metrics['expectancy'] = float(m.group(1))
    # Extract monthly lines
    monthly = []
    for line in output.split('\n'):
        if re.match(r'\s*\d{4}-\d{2}\s*\|', line):
            monthly.append(line.strip())
    metrics['monthly'] = monthly
    return metrics

# Config override for combined: match CB original exactly
COMBINED_OVERRIDES = """
import backtest_combined_mr_breakout as m

# Align to CB original
m.ENABLE_MR = False
m.ENABLE_CB = True
m.RISK_PERCENT = 0.01
m.WAIT_CANDLES = 3
m.DISPLAY_MODE = "MONTHLY"

# CB params matching original
m.CB_MIN_RR = 2.0
m.CB_SL_OFFSET = 1.0
m.CB_TP1_RR = 1.0
m.CB_TP1_SPLIT = 0.3
m.CB_TP2_SPLIT = 0.7
m.CB_USE_TRAILING = True
m.CB_USE_VP_STRUCTURE_FILTER = True
m.CB_MIN_POC_STRENGTH = 3.0
m.CB_EXCLUDE_VAH_TARGET = True
m.CB_USE_PREV_DAY = True
m.CB_USE_PREV_WEEK = True
m.CB_EXCLUDED_HOURS = [0, 10]

# Match ASSETS exactly
m.ASSETS = [
    {
        'enabled': True,
        'symbol': 'XAUUSD',
        'candle_table': 'candles_mt5_xauusd_1m',
        'tick_table': 'market_ticks_xauusd',
        'tick_size': 0.01,
        'va_percent': 0.70,
        'allow_long': True,
        'allow_short': True,
        'sessions': {'TOKYO': True, 'LONDON': True, 'NY': False},
        'allowed_days': [0, 1, 2, 3],
    },
]

m.run_backtest()
"""

print("=" * 80)
print("RUNNING CB ORIGINAL...")
print("=" * 80)
r1 = subprocess.run(
    [sys.executable, 'backtest_confirmed_breakout.py'],
    capture_output=True, text=True, cwd='.', env=ENV
)
out_cb_orig = r1.stdout + r1.stderr
print(out_cb_orig[-3000:] if len(out_cb_orig) > 3000 else out_cb_orig)

print("\n" + "=" * 80)
print("RUNNING COMBINED (CB-ONLY, aligned config)...")
print("=" * 80)
r2 = subprocess.run(
    [sys.executable, '-c', COMBINED_OVERRIDES],
    capture_output=True, text=True, cwd='.', env=ENV
)
out_cb_comb = r2.stdout + r2.stderr
print(out_cb_comb[-3000:] if len(out_cb_comb) > 3000 else out_cb_comb)

# Compare
m1 = extract_metrics(out_cb_orig)
m2 = extract_metrics(out_cb_comb)

print("\n" + "=" * 80)
print("COMPARISON: CB Original vs Combined (CB-only)")
print("=" * 80)
print(f"{'Metric':<20} | {'CB Original':>15} | {'Combined CB':>15} | {'Match':>6}")
print("-" * 65)
for key in ['trades', 'wins', 'be', 'losses', 'wr', 'pnl_r', 'pf', 'capital', 'dd', 'expectancy']:
    v1 = m1.get(key, 'N/A')
    v2 = m2.get(key, 'N/A')
    if isinstance(v1, float) and isinstance(v2, float):
        match = "OK" if abs(v1 - v2) < 0.01 else "DIFF"
        print(f"  {key:<18} | {v1:>15.2f} | {v2:>15.2f} | {match:>6}")
    elif isinstance(v1, int) and isinstance(v2, int):
        match = "OK" if v1 == v2 else "DIFF"
        print(f"  {key:<18} | {v1:>15} | {v2:>15} | {match:>6}")
    else:
        print(f"  {key:<18} | {str(v1):>15} | {str(v2):>15} |      ?")

# Compare monthly
print("\n" + "-" * 65)
print("MONTHLY COMPARISON")
print("-" * 65)
ml1 = m1.get('monthly', [])
ml2 = m2.get('monthly', [])
max_len = max(len(ml1), len(ml2))
for i in range(max_len):
    l1 = ml1[i] if i < len(ml1) else "(missing)"
    l2 = ml2[i] if i < len(ml2) else "(missing)"
    tag = "OK" if l1 == l2 else "DIFF"
    if tag == "DIFF":
        print(f"  [{tag}] ORIG: {l1}")
        print(f"        COMB: {l2}")
    else:
        print(f"  [{tag}] {l1}")
