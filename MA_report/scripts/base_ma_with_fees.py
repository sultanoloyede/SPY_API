# ma_default_costs.py
# Moving-average crossover backtest (no ATR, no stops).
# - Buy when FAST crosses above SLOW; sell/short when FAST crosses below SLOW
# - Always one position: a new signal exits old trade and enters new one at NEXT bar's open
# - Costs: $25 commission + $5 slippage PER ROUND TURN (deducted when a trade closes)
# Outputs: trades.csv, ma_cross_results.csv, equity_curve.png, drawdown_percent.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======== CONFIG ========
CSV_PATH = "../candlesticks/EURUSD_30m.csv"
FAST = 10
SLOW = 30

INITIAL_CAPITAL = 100_000.0
RISK_DOLLARS = 1_000.0      # exposure normalization only (no stop)
RISK_MOVE_PCT = 0.01        # assume 1% move ~ $1k P&L for unit sizing

COMMISSION_PER_RT = 25.0    # per round turn (entry+exit)
SLIPPAGE_PER_RT  = 5.0      # per round turn
COST_PER_RT = COMMISSION_PER_RT + SLIPPAGE_PER_RT

OUT_TRADES   = "../metrics/trades_fees.csv"
OUT_SUMMARY  = "../metrics/ma_cross_results_fees.csv"
OUT_EQUITY   = "../graphics/equity_curve_fees.png"
OUT_DD_PCT   = "../graphics/drawdown_percent_fees.png"

# ======== IO / LOADING ========
def _match(df, names):
    for n in names:
        if n in df.columns: return n
        for c in df.columns:
            if c.lower() == n.lower(): return c
    return None

def load_data(path):
    df = pd.read_csv(path)
    dt = _match(df, ["datetime","timestamp","time","date"])
    if dt is None:
        raise ValueError("CSV must contain a datetime-like column (e.g., 'datetime').")
    df[dt] = pd.to_datetime(df[dt])
    df = df.sort_values(dt).reset_index(drop=True)

    o = _match(df, ["open"])
    h = _match(df, ["high"])
    l = _match(df, ["low"])
    c = _match(df, ["close","adj close"])
    if c is None: raise ValueError("CSV must contain a 'close' column.")

    df = df.rename(columns={dt: "datetime"})
    df["open"]  = df[o] if o else df[c]
    df["high"]  = df[h] if h else df[c]
    df["low"]   = df[l] if l else df[c]
    df["close"] = df[c]
    return df[["datetime","open","high","low","close"]]

# ======== INDICATORS / SIGNALS ========
def with_signals(df, fast=FAST, slow=SLOW):
    df["fast"] = df["close"].rolling(fast).mean()
    df["slow"] = df["close"].rolling(slow).mean()

    valid = df["fast"].notna() & df["slow"].notna()
    prev_fast = df["fast"].shift(1)
    prev_slow = df["slow"].shift(1)

    cross_up   = (prev_fast <= prev_slow) & (df["fast"] > df["slow"])
    cross_down = (prev_fast >= prev_slow) & (df["fast"] < df["slow"])

    df["signal"] = np.where(cross_up, 1, np.where(cross_down, -1, 0))
    df.loc[~valid, "signal"] = 0
    return df

# ======== BACKTEST (always in; reverse on signal) ========
def backtest(df):
    cash = INITIAL_CAPITAL
    direction = 0        # 1 long, -1 short, 0 flat
    units = 0.0
    entry_price = np.nan
    entry_idx = None

    equity_times, equity_vals = [], []
    trades = []

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]

        # mark-to-market on current close
        mtm = cash + (units * (row["close"] - entry_price) if direction != 0 else 0.0)
        equity_times.append(row["datetime"])
        equity_vals.append(mtm)

        if row["signal"] != 0:
            # 1) Close old position at next open (deduct the whole round-turn cost now)
            if direction != 0:
                exit_price = nxt["open"]
                gross_pnl = units * (exit_price - entry_price)
                net_pnl   = gross_pnl - COST_PER_RT
                cash += net_pnl
                trades.append({
                    "entry_time": df.iloc[entry_idx]["datetime"],
                    "exit_time":  nxt["datetime"],
                    "side": "LONG" if direction==1 else "SHORT",
                    "units": units,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "bars_held": i - entry_idx + 1,
                    "gross_pnl": gross_pnl,
                    "costs_rt": COST_PER_RT,
                    "pnl": net_pnl,
                })
                direction = 0; units = 0.0; entry_price = np.nan; entry_idx = None

            # 2) Open new position at next open
            direction = int(row["signal"])
            assumed_move = max(RISK_MOVE_PCT * row["close"], 1e-12)
            size_units = RISK_DOLLARS / assumed_move
            units = direction * size_units
            entry_price = nxt["open"]
            entry_idx = i+1

    # close any open position on the last bar (also charge the round-turn cost)
    last = df.iloc[-1]
    if direction != 0:
        exit_price = last["close"]
        gross_pnl = units * (exit_price - entry_price)
        net_pnl   = gross_pnl - COST_PER_RT
        cash += net_pnl
        trades.append({
            "entry_time": df.iloc[entry_idx]["datetime"],
            "exit_time":  last["datetime"],
            "side": "LONG" if direction==1 else "SHORT",
            "units": units,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bars_held": len(df)-1 - entry_idx + 1,
            "gross_pnl": gross_pnl,
            "costs_rt": COST_PER_RT,
            "pnl": net_pnl,
        })

    equity_times.append(last["datetime"])
    equity_vals.append(cash)

    equity = pd.Series(equity_vals, index=pd.to_datetime(equity_times), name="equity")
    trades = pd.DataFrame(trades)
    return equity, trades

# ======== METRICS / TABLE ========
def max_drawdown_amount(equity: pd.Series):
    peak = equity.cummax()
    dd = equity - peak
    mdd_amt = -dd.min() if len(dd) else 0.0   # positive dollars
    mdd_date = dd.idxmin() if len(dd) else pd.NaT
    return mdd_amt, mdd_date

def drawdown_percent_series(equity: pd.Series):
    peak = equity.cummax()
    dd_pct = equity / peak - 1.0
    return dd_pct  # e.g., -0.123 = -12.3%

def _streaks(bools):
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best

def _block(tr_df: pd.DataFrame):
    pnl = tr_df["pnl"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return {
        "Total Net Profit": pnl.sum(),
        "Gross Profit": wins.sum(),
        "Gross Loss": losses.sum(),
        "Profit Factor": (wins.sum()/abs(losses.sum())) if losses.sum()!=0 else np.nan,
        "Total Number of Trades": len(tr_df),
        "Percent Profitable": (len(wins)/len(tr_df)*100.0) if len(tr_df) else np.nan,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Avg. Trade Net Profit": pnl.mean() if len(tr_df) else np.nan,
        "Avg. Winning Trade": wins.mean() if len(wins) else np.nan,
        "Avg. Losing Trade": losses.mean() if len(losses) else np.nan,
        "Ratio Avg. Win:Avg. Loss": (wins.mean()/abs(losses.mean())) if len(wins) and len(losses) else np.nan,
        "Largest Winning Trade": wins.max() if len(wins) else np.nan,
        "Largest Losing Trade": losses.min() if len(losses) else np.nan,
        "Max. Consecutive Winning Trades": _streaks([x>0 for x in pnl]),
        "Max. Consecutive Losing Trades": _streaks([x<=0 for x in pnl]),
        "Avg. Bars in Total Trades": tr_df["bars_held"].mean() if len(tr_df) else np.nan,
        "Avg. Bars in Winning Trades": tr_df[tr_df["pnl"]>0]["bars_held"].mean() if len(wins) else np.nan,
        "Avg. Bars in Losing Trades": tr_df[tr_df["pnl"]<=0]["bars_held"].mean() if len(losses) else np.nan,
    }

def _money_fmt(x):
    if isinstance(x, (int,float,np.integer,np.floating)) and not pd.isna(x):
        return f"${abs(x):,.0f}" if x >= 0 else f"(${abs(x):,.0f})"
    return "" if (x is None or (isinstance(x,float) and pd.isna(x))) else str(x)

def _num_fmt(x):
    if isinstance(x, (int,float,np.integer,np.floating)) and not pd.isna(x):
        return f"{x:.2f}"
    return ""

def build_table(trades: pd.DataFrame, equity: pd.Series):
    all_blk   = _block(trades)
    long_blk  = _block(trades[trades["side"]=="LONG"])
    short_blk = _block(trades[trades["side"]=="SHORT"])

    mdd_amt, mdd_date = max_drawdown_amount(equity)
    for b in (all_blk, long_blk, short_blk):
        # store negative so formatter prints parentheses like in your screenshot
        b["Max. Drawdown (Intraday Peak to Valley)"] = -mdd_amt
        b["Date of Max. Drawdown"] = mdd_date.strftime("%d-%b-%y") if pd.notna(mdd_date) else ""

    rows = list(all_blk.keys())
    out = pd.DataFrame(
        {
            "All Trades":   [all_blk[r]   for r in rows],
            "Long Trades":  [long_blk[r]  for r in rows],
            "Short Trades": [short_blk[r] for r in rows],
        },
        index=rows,
    )

    # currency rows only (avoid formatting the date)
    currency_rows = [
        "Total Net Profit","Gross Profit","Gross Loss",
        "Avg. Trade Net Profit","Avg. Winning Trade","Avg. Losing Trade",
        "Largest Winning Trade","Largest Losing Trade",
        "Max. Drawdown (Intraday Peak to Valley)",
    ]
    for col in out.columns:
        out.loc[currency_rows, col] = out.loc[currency_rows, col].apply(_money_fmt)

    out.loc["Profit Factor"]      = out.loc["Profit Factor"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    out.loc["Percent Profitable"] = out.loc["Percent Profitable"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.2f}%")
    for r in ["Avg. Bars in Total Trades","Avg. Bars in Winning Trades","Avg. Bars in Losing Trades"]:
        out.loc[r] = out.loc[r].apply(_num_fmt)
    return out

# ======== MAIN ========
def main():
    df = load_data(CSV_PATH)
    df = with_signals(df, FAST, SLOW)
    equity, trades = backtest(df)

    # Save blotter
    trades.to_csv(OUT_TRADES, index=False)

    # Equity curve
    plt.figure(figsize=(10,4))
    plt.plot(equity.index, equity.values)
    plt.title("Equity Curve (Detailed) — MA Crossover (with $30 RT costs)")
    plt.xlabel("Time"); plt.ylabel("Equity ($)")
    plt.tight_layout(); plt.savefig(OUT_EQUITY, dpi=150); plt.close()

    # Drawdown % (filled area, like your example)
    dd_pct = drawdown_percent_series(equity) * 100.0
    plt.figure(figsize=(10,4))
    plt.fill_between(dd_pct.index, dd_pct.values, 0)   # keep default style (no explicit colors)
    plt.title("Drawdown (%)")
    plt.xlabel("Time"); plt.ylabel("Drawdown (%)")
    plt.tight_layout(); plt.savefig(OUT_DD_PCT, dpi=150); plt.close()

    # Summary table
    summary = build_table(trades, equity)
    summary.to_csv(OUT_SUMMARY)

    print(f"Saved: {OUT_TRADES}, {OUT_EQUITY}, {OUT_DD_PCT}, {OUT_SUMMARY}")

if __name__ == "__main__":
    main()