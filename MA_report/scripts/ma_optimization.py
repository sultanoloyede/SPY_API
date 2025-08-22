# ma_default_grid_costs_fixed.py
# MA crossover backtest with costs + parameter sweep + 3D surface.
# Robust to "no-trade" parameter picks.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from itertools import product

# ======== CONFIG ========
CSV_PATH = "../candlesticks/EURUSD_30m.csv"

INITIAL_CAPITAL = 100_000.0
RISK_DOLLARS = 1_000.0          # exposure normalization only (no stop)
RISK_MOVE_PCT = 0.01            # 1% move ≈ $1k P&L for unit sizing

COMMISSION_PER_RT = 25.0
SLIPPAGE_PER_RT  = 5.0
COST_PER_RT = COMMISSION_PER_RT + SLIPPAGE_PER_RT  # $30 per round turn

SLOW_RANGE = list(range(80, 121, 2))   # 20,24,...,80
FAST_RANGE = list(range(25, 46, 1))    # 0..20 ; 0/1 => price
REQUIRE_MIN_TRADES = 1                # choose best params with at least this many trades

SHOW_PLOTS = True                     # also call plt.show() after saving PNGs

OUT_TRADES   = "../metrics/trades_para.csv"
OUT_SUMMARY  = "../metrics/ma_cross_results_para.csv"
OUT_EQUITY   = "../graphics/equity_curve_para.png"
OUT_DD_PCT   = "../graphics/drawdown_percent_para.png"
OUT_SURFACE  = "../graphics/ma_grid_surface_para.png"
OUT_GRIDCSV  = "../metrics/ma_grid_results_para.csv"

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

# ======== SIGNALS ========
def rolling_sma(series: pd.Series, n: int):
    if n <= 1:
        return series.copy()
    return series.rolling(n).mean()

def with_signals(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    out = df.copy()
    out["fast"] = rolling_sma(out["close"], fast)
    out["slow"] = rolling_sma(out["close"], slow)

    valid = out["fast"].notna() & out["slow"].notna()
    pf, ps = out["fast"].shift(1), out["slow"].shift(1)

    cross_up   = (pf <= ps) & (out["fast"] > out["slow"])
    cross_down = (pf >= ps) & (out["fast"] < out["slow"])
    out["signal"] = np.where(cross_up, 1, np.where(cross_down, -1, 0))
    out.loc[~valid, "signal"] = 0
    return out

# ======== BACKTEST (always in; reverse on signal) ========
TRADE_COLS = [
    "entry_time","exit_time","side","units",
    "entry_price","exit_price","bars_held",
    "gross_pnl","costs_rt","pnl"
]

def backtest(df: pd.DataFrame):
    cash = INITIAL_CAPITAL
    direction = 0
    units = 0.0
    entry_price = np.nan
    entry_idx = None

    equity_times, equity_vals = [], []
    trades_list = []

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]

        mtm = cash + (units * (row["close"] - entry_price) if direction != 0 else 0.0)
        equity_times.append(row["datetime"])
        equity_vals.append(mtm)

        if row["signal"] != 0:
            if direction != 0:
                exit_price = nxt["open"]
                gross_pnl = units * (exit_price - entry_price)
                net_pnl   = gross_pnl - COST_PER_RT
                cash += net_pnl
                trades_list.append({
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

            direction = int(row["signal"])
            assumed_move = max(RISK_MOVE_PCT * row["close"], 1e-12)
            size_units = RISK_DOLLARS / assumed_move
            units = direction * size_units
            entry_price = nxt["open"]
            entry_idx = i+1

    last = df.iloc[-1]
    if direction != 0:
        exit_price = last["close"]
        gross_pnl = units * (exit_price - entry_price)
        net_pnl   = gross_pnl - COST_PER_RT
        cash += net_pnl
        trades_list.append({
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
    # ensure required columns exist even if there are no trades
    trades = pd.DataFrame(trades_list, columns=TRADE_COLS)
    return equity, trades

# ======== METRICS / TABLE ========
def max_drawdown_amount(equity: pd.Series):
    peak = equity.cummax()
    dd = equity - peak
    mdd_amt = -dd.min() if len(dd) else 0.0
    mdd_date = dd.idxmin() if len(dd) else pd.NaT
    return mdd_amt, mdd_date

def drawdown_percent_series(equity: pd.Series):
    peak = equity.cummax()
    return equity/peak - 1.0

def _streaks(bools):
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best

def _block(tr_df: pd.DataFrame):
    pnl = tr_df["pnl"] if "pnl" in tr_df.columns else pd.Series(dtype=float)
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    return {
        "Total Net Profit": pnl.sum() if len(pnl) else 0.0,
        "Gross Profit": wins.sum() if len(wins) else 0.0,
        "Gross Loss": losses.sum() if len(losses) else 0.0,
        "Profit Factor": (wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else np.nan,
        "Total Number of Trades": len(tr_df),
        "Percent Profitable": (len(wins)/len(tr_df)*100.0) if len(tr_df) else np.nan,
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Avg. Trade Net Profit": pnl.mean() if len(pnl) else np.nan,
        "Avg. Winning Trade": wins.mean() if len(wins) else np.nan,
        "Avg. Losing Trade": losses.mean() if len(losses) else np.nan,
        "Ratio Avg. Win:Avg. Loss": (wins.mean()/abs(losses.mean())) if len(wins) and len(losses) else np.nan,
        "Largest Winning Trade": wins.max() if len(wins) else np.nan,
        "Largest Losing Trade": losses.min() if len(losses) else np.nan,
        "Max. Consecutive Winning Trades": _streaks([x>0 for x in pnl]) if len(pnl) else 0,
        "Max. Consecutive Losing Trades": _streaks([x<=0 for x in pnl]) if len(pnl) else 0,
        "Avg. Bars in Total Trades": tr_df["bars_held"].mean() if len(tr_df) else np.nan,
        "Avg. Bars in Winning Trades": tr_df[tr_df.get("pnl", pd.Series(dtype=float))>0]["bars_held"].mean() if len(wins) else np.nan,
        "Avg. Bars in Losing Trades": tr_df[tr_df.get("pnl", pd.Series(dtype=float))<=0]["bars_held"].mean() if len(losses) else np.nan,
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
    long_blk  = _block(trades[trades["side"]=="LONG"]) if "side" in trades.columns else _block(pd.DataFrame(columns=TRADE_COLS))
    short_blk = _block(trades[trades["side"]=="SHORT"]) if "side" in trades.columns else _block(pd.DataFrame(columns=TRADE_COLS))

    mdd_amt, mdd_date = max_drawdown_amount(equity)
    for b in (all_blk, long_blk, short_blk):
        b["Max. Drawdown (Intraday Peak to Valley)"] = -mdd_amt
        b["Date of Max. Drawdown"] = mdd_date.strftime("%d-%b-%y") if pd.notna(mdd_date) else ""

    rows = list(all_blk.keys())
    out = pd.DataFrame(
        {"All Trades":[all_blk[r] for r in rows],
         "Long Trades":[long_blk[r] for r in rows],
         "Short Trades":[short_blk[r] for r in rows]},
        index=rows,
    )

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

# ======== GRID SEARCH + PLOTS ========
def grid_search(df: pd.DataFrame):
    results = []
    for slow, fast in product(SLOW_RANGE, FAST_RANGE):
        sig = with_signals(df, fast, slow)
        eq, tr = backtest(sig)
        end_eq = float(eq.iloc[-1]) if len(eq) else INITIAL_CAPITAL
        n_tr   = len(tr)
        results.append((slow, fast, end_eq, n_tr))
    grid = pd.DataFrame(results, columns=["slow","fast","end_equity","n_trades"])
    return grid

def plot_surface(grid: pd.DataFrame, path_png: str, show=SHOW_PLOTS):
    slows = sorted(grid["slow"].unique())
    fasts = sorted(grid["fast"].unique())
    X, Y = np.meshgrid(slows, fasts)
    Z = np.zeros_like(X, dtype=float)
    map_end = {(int(r.slow), int(r.fast)): r.end_equity for r in grid.itertuples()}
    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = map_end.get((int(X[i,j]), int(Y[i,j])), np.nan)

    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    ax.set_xlabel("Slow MA")
    ax.set_ylabel("Fast MA")
    ax.set_zlabel("Ending Equity ($)")
    ax.set_title("MA Grid Search — Ending Equity")
    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

def main():
    df = load_data(CSV_PATH)

    # 1) Grid search
    grid = grid_search(df)
    grid.to_csv(OUT_GRIDCSV, index=False)
    plot_surface(grid, OUT_SURFACE, show=SHOW_PLOTS)

    # 2) Choose best params with trades; fallback to best equity if none
    grid_w_trades = grid[grid["n_trades"] >= REQUIRE_MIN_TRADES]
    if len(grid_w_trades):
        best = grid_w_trades.sort_values("end_equity", ascending=False).iloc[0]
    else:
        best = grid.sort_values("end_equity", ascending=False).iloc[0]
    best_slow, best_fast = int(best.slow), int(best.fast)
    print(f"Best by ending equity (min trades={REQUIRE_MIN_TRADES}) -> "
          f"SLOW={best_slow}, FAST={best_fast}, trades={int(best.n_trades)}, End Equity=${best.end_equity:,.0f}")

    # 3) Final backtest & outputs
    sig = with_signals(df, best_fast, best_slow)
    equity, trades = backtest(sig)

    trades.to_csv(OUT_TRADES, index=False)

    plt.figure(figsize=(10,4))
    plt.plot(equity.index, equity.values)
    plt.title(f"Equity Curve — Slow={best_slow}, Fast={best_fast} (with $30 RT costs)")
    plt.xlabel("Time"); plt.ylabel("Equity ($)")
    plt.tight_layout(); plt.savefig(OUT_EQUITY, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    dd_pct = drawdown_percent_series(equity) * 100.0
    plt.figure(figsize=(10,4))
    plt.fill_between(dd_pct.index, dd_pct.values, 0)
    plt.title("Drawdown (%)")
    plt.xlabel("Time"); plt.ylabel("Drawdown (%)")
    plt.tight_layout(); plt.savefig(OUT_DD_PCT, dpi=150)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    summary = build_table(trades, equity)
    summary.to_csv(OUT_SUMMARY)

    print(f"Saved: {OUT_GRIDCSV}, {OUT_SURFACE}, {OUT_TRADES}, {OUT_EQUITY}, {OUT_DD_PCT}, {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
