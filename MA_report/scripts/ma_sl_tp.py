import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import calendar

# ========= CONFIG =========
CSV_PATH = "../candlesticks/EURUSD_30m.csv"

FAST = 33
SLOW = 98

TRADE_WINDOW_START_HOUR = 14
TRADE_WINDOW_START_MINUTE = 30
TRADE_WINDOW_LENGTH_HOURS = 4 # inclusive window length

INITIAL_CAPITAL   = 100_000.0
RISK_DOLLARS      = 1_000.0          # sizing anchor: dollars per RISK_MOVE_PCT price move
RISK_MOVE_PCT     = 0.01              # 10% move ↔ RISK_DOLLARS P&L for unit sizing
COMMISSION_PER_RT = 25.0
SLIPPAGE_PER_RT   = 5.0
COST_PER_RT       = COMMISSION_PER_RT + SLIPPAGE_PER_RT  # $30 per round turn

STOP_LOSS_PCT     = 0.01             # 1.0% protective stop
USE_STOP_LOSS     = False              # <<< toggle stop-loss ON/OFF

# outputs
OUT_TRADES                    = "../metrics/trades_sl_tp.csv"
OUT_EQUITY                    = "../graphics/equity_curve_sl_tp.png"
OUT_DRAWDOWN_PCT              = "../graphics/drawdown_percent_sl_tp.png"
OUT_SCATTER_DOLLARS           = "../graphics/profit_vs_drawdown_sl_tp.png"
OUT_SCATTER_PERCENTAGES       = "../graphics/profit_vs_drawdown_pct_sl_tp.png"
OUT_SCATTER_PERCENTAGES_ZOOM  = "../graphics/profit_vs_drawdown_pct_zoom_sl_tp.png"
OUT_MONTHLY_AVG_PROFIT        = "../graphics/monthly_avg_profit_sl_tp.png"
OUT_PERFORMANCE_CSV           = "../metrics/performance_report_sl_tp.csv"
SHOW_PLOTS = True

# ========= UTIL =========
def ensure_dir_for(file_path: str):
    d = os.path.dirname(file_path)
    if d:
        os.makedirs(d, exist_ok=True)

def _match(df, names):
    for n in names:
        if n in df.columns: return n
        for c in df.columns:
            if c.lower() == n.lower(): return c
    return None

def stop_suffix():
    return f" + {STOP_LOSS_PCT*100:.1f}% Stop" if USE_STOP_LOSS else ""

# ========= IO / SIGNALS =========
def within_trade_window(dt):
    start = dt.replace(hour=TRADE_WINDOW_START_HOUR, minute=TRADE_WINDOW_START_MINUTE, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=TRADE_WINDOW_LENGTH_HOURS)
    return start <= dt <= end

def load_data(path):
    df = pd.read_csv(path)
    dt = _match(df, ["datetime","timestamp","time","date"])
    if dt is None:
        raise ValueError("CSV must contain a datetime-like column (e.g., 'datetime').")
    df[dt] = pd.to_datetime(df[dt])
    df = df.sort_values(dt).reset_index(drop=True)

    o = _match(df, ["open"]); h = _match(df, ["high"])
    l = _match(df, ["low"]);  c = _match(df, ["close","adj close"])
    if c is None:
        raise ValueError("CSV must contain a 'close' column.")

    df = df.rename(columns={dt: "datetime"})
    df["open"]  = df[o] if o else df[c]
    df["high"]  = df[h] if h else df[c]
    df["low"]   = df[l] if l else df[c]
    df["close"] = df[c]
    return df[["datetime","open","high","low","close"]]

def rolling_sma(series: pd.Series, n: int):
    if n <= 1: return series.copy()
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

# ========= BACKTEST WITH OPTIONAL STOP =========
TRADE_COLS = [
    "entry_time","exit_time","side","units",
    "entry_price","exit_price","bars_held",
    "gross_pnl","costs_rt","pnl",
    "entry_idx","exit_idx","exit_reason"
]

def backtest_with_stop(df: pd.DataFrame, stop_loss_pct: float, use_stop_loss: bool):
    cash = INITIAL_CAPITAL
    direction = 0                 # +1 long, -1 short, 0 flat
    units = 0.0
    entry_price = np.nan
    entry_idx = None
    stop_price = np.nan

    equity_times, equity_vals = [], []
    trades_list = []

    for i in range(1, len(df)-1):
        row = df.iloc[i]
        nxt = df.iloc[i+1]
        stopped_this_bar = False

        # 1) Intrabar protective stop check (only if enabled)
        if use_stop_loss and direction != 0:
            if direction == 1 and row["low"] <= stop_price:
                exit_price = stop_price
                gross_pnl = units * (exit_price - entry_price)
                net_pnl   = gross_pnl - COST_PER_RT
                cash += net_pnl
                trades_list.append({
                    "entry_time": df.iloc[entry_idx]["datetime"],
                    "exit_time":  row["datetime"],
                    "side": "LONG",
                    "units": units,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "bars_held": i - entry_idx + 1,
                    "gross_pnl": gross_pnl,
                    "costs_rt": COST_PER_RT,
                    "pnl": net_pnl,
                    "entry_idx": entry_idx,
                    "exit_idx":  i,
                    "exit_reason": "stop"
                })
                direction = 0; units = 0.0; entry_price = np.nan; entry_idx = None
                stopped_this_bar = True

            if direction == -1 and row["high"] >= stop_price:
                exit_price = stop_price
                gross_pnl = units * (exit_price - entry_price)
                net_pnl   = gross_pnl - COST_PER_RT
                cash += net_pnl
                trades_list.append({
                    "entry_time": df.iloc[entry_idx]["datetime"],
                    "exit_time":  row["datetime"],
                    "side": "SHORT",
                    "units": units,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "bars_held": i - entry_idx + 1,
                    "gross_pnl": gross_pnl,
                    "costs_rt": COST_PER_RT,
                    "pnl": net_pnl,
                    "entry_idx": entry_idx,
                    "exit_idx":  i,
                    "exit_reason": "stop"
                })
                direction = 0; units = 0.0; entry_price = np.nan; entry_idx = None
                stopped_this_bar = True

        # 2) Mark-to-market equity
        mtm = cash + (units * (row["close"] - entry_price) if direction != 0 else 0.0)
        equity_times.append(row["datetime"])
        equity_vals.append(mtm)

        # 3) Signal processing (reverse on signal at next bar's open)
        if row["signal"] != 0 and within_trade_window(row["datetime"]):
            sig_dir = int(row["signal"])

            if direction != 0 and not stopped_this_bar:
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
                    "entry_idx": entry_idx,
                    "exit_idx":  i+1,
                    "exit_reason": "signal"
                })
                direction = 0; units = 0.0; entry_price = np.nan; entry_idx = None

            # enter (or re-enter) at next open in signal direction
            assumed_move = max(RISK_MOVE_PCT * row["close"], 1e-12)
            size_units = RISK_DOLLARS / assumed_move
            direction = sig_dir
            units = direction * size_units
            entry_price = nxt["open"]
            entry_idx = i+1
            if use_stop_loss:
                stop_price = (entry_price * (1.0 - stop_loss_pct)) if direction == 1 else (entry_price * (1.0 + stop_loss_pct))
            else:
                stop_price = np.nan

    # 4) Liquidate on final bar close
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
            "entry_idx": entry_idx,
            "exit_idx":  len(df)-1,
            "exit_reason": "liquidate"
        })

    equity_times.append(last["datetime"])
    equity_vals.append(cash)
    equity = pd.Series(equity_vals, index=pd.to_datetime(equity_times), name="equity")
    trades = pd.DataFrame(trades_list, columns=TRADE_COLS)
    return equity, trades

# ========= METRICS =========
def drawdown_percent_series(equity: pd.Series):
    peak = equity.cummax()
    return equity/peak - 1.0

def max_drawdown_amount(equity: pd.Series):
    peak = equity.cummax()
    dd = equity - peak
    mdd_amt = -dd.min() if len(dd) else 0.0
    mdd_date = dd.idxmin() if len(dd) else pd.NaT
    return mdd_amt, mdd_date

def _streaks(bools):
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best

def compute_trade_mae_dollars(df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    maes = []
    for r in trades.itertuples():
        sl = df.iloc[r.entry_idx:r.exit_idx+1]
        if r.side == "LONG":
            worst_price = sl["low"].min()
            adverse_move = max(0.0, r.entry_price - worst_price)
        else:
            worst_price = sl["high"].max()
            adverse_move = max(0.0, worst_price - r.entry_price)
        maes.append(abs(r.units) * adverse_move)
    return pd.Series(maes, index=trades.index, name="mae_dollars")

def compute_trade_mae_pct(df: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    maes_pct = []
    for r in trades.itertuples():
        sl = df.iloc[r.entry_idx:r.exit_idx+1]
        if r.side == "LONG":
            worst_price = sl["low"].min(); dd = max(0.0, (r.entry_price - worst_price) / r.entry_price)
        else:
            worst_price = sl["high"].max(); dd = max(0.0, (worst_price - r.entry_price) / r.entry_price)
        maes_pct.append(dd * 100.0)
    return pd.Series(maes_pct, index=trades.index, name="mae_pct")

def compute_trade_return_pct(trades: pd.DataFrame) -> pd.Series:
    rets = []
    for r in trades.itertuples():
        move = (r.exit_price - r.entry_price) / r.entry_price
        signed = move if r.side == "LONG" else -move
        rets.append(signed * 100.0)  # price-based returns
    return pd.Series(rets, index=trades.index, name="ret_pct")

# ---------- Performance table (All / Long / Short) ----------
def _money_fmt(x):
    if pd.isna(x): return ""
    return f"${abs(x):,.0f}" if x >= 0 else f"(${abs(x):,.0f})"

def _num_fmt(x, n=2):
    if pd.isna(x): return ""
    return f"{float(x):.{n}f}"

def build_performance_table(trades: pd.DataFrame, equity: pd.Series, n_total_bars: int) -> pd.DataFrame:
    def block(tr_df: pd.DataFrame):
        pnl = tr_df["pnl"] if "pnl" in tr_df.columns else pd.Series(dtype=float)
        wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
        total_trades = len(tr_df)

        out = {
            "Total Net Profit": pnl.sum() if len(pnl) else 0.0,
            "Gross Profit": wins.sum() if len(wins) else 0.0,
            "Gross Loss": losses.sum() if len(losses) else 0.0,
            "Profit Factor": (wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else np.nan,
            "Total Number of Trades": total_trades,
            "Percent Profitable": (len(wins)/total_trades*100.0) if total_trades else np.nan,
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
            # Costs
            "Total Slippage": len(tr_df) * SLIPPAGE_PER_RT,
            "Total Commission": len(tr_df) * COMMISSION_PER_RT,
            # Time in market
            "Percent of Time in the Market": (tr_df["bars_held"].sum()/n_total_bars*100.0) if n_total_bars>0 else np.nan,
        }
        return out

    all_blk   = block(trades)
    long_blk  = block(trades[trades["side"]=="LONG"])
    short_blk = block(trades[trades["side"]=="SHORT"])

    mdd_amt, mdd_date = max_drawdown_amount(equity)
    mdd_row = "Max. Drawdown (Intraday Peak to Valley)"
    mdd_date_row = "Date of Max. Drawdown"
    for b in (all_blk, long_blk, short_blk):
        b[mdd_row] = -mdd_amt    # negative number meaning drawdown
        b[mdd_date_row] = mdd_date.strftime("%d-%b-%y") if pd.notna(mdd_date) else ""

    # Assemble table
    rows = [
        "Total Net Profit","Gross Profit","Gross Loss","Profit Factor",
        "Total Number of Trades","Percent Profitable","Winning Trades","Losing Trades",
        "Avg. Trade Net Profit","Avg. Winning Trade","Avg. Losing Trade","Ratio Avg. Win:Avg. Loss",
        "Largest Winning Trade","Largest Losing Trade",
        "Max. Consecutive Winning Trades","Max. Consecutive Losing Trades",
        "Avg. Bars in Total Trades","Avg. Bars in Winning Trades","Avg. Bars in Losing Trades",
        "Total Slippage","Total Commission","Percent of Time in the Market",
        mdd_row, mdd_date_row,
    ]
    table = pd.DataFrame(
        {"All Trades":[all_blk[r] for r in rows],
         "Long Trades":[long_blk[r] for r in rows],
         "Short Trades":[short_blk[r] for r in rows]},
        index=rows,
    )

    # Pretty formatting to match screenshot style
    currency_rows = [
        "Total Net Profit","Gross Profit","Gross Loss",
        "Avg. Trade Net Profit","Avg. Winning Trade","Avg. Losing Trade",
        "Largest Winning Trade","Largest Losing Trade",
        "Total Slippage","Total Commission",
        mdd_row,
    ]
    for col in table.columns:
        table.loc[currency_rows, col] = table.loc[currency_rows, col].apply(_money_fmt)
    table.loc["Profit Factor"]      = table.loc["Profit Factor"].apply(lambda x: "" if pd.isna(x) else _num_fmt(x,2))
    table.loc["Percent Profitable"] = table.loc["Percent Profitable"].apply(lambda x: "" if pd.isna(x) else f"{_num_fmt(x,2)}%")
    table.loc["Percent of Time in the Market"] = table.loc["Percent of Time in the Market"].apply(lambda x: "" if pd.isna(x) else f"{_num_fmt(x,2)}%")
    for r in ["Avg. Bars in Total Trades","Avg. Bars in Winning Trades","Avg. Bars in Losing Trades"]:
        table.loc[r] = table.loc[r].apply(lambda x: "" if pd.isna(x) else _num_fmt(x,2))

    return table

# ========= PLOTS =========
def plot_equity_and_drawdown(equity: pd.Series, out_equity: str, out_dd_pct: str, show=True):
    ensure_dir_for(out_equity); ensure_dir_for(out_dd_pct)
    # Equity
    plt.figure(figsize=(10,4))
    plt.plot(equity.index, equity.values, linewidth=1.2)
    plt.title(f"Equity Curve — MA(33,98){stop_suffix()}")
    plt.xlabel("Time"); plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.savefig(out_equity, dpi=150)
    if show: plt.show()
    plt.close()

    # Drawdown %
    dd_pct = drawdown_percent_series(equity) * 100.0
    plt.figure(figsize=(10,4))
    plt.fill_between(dd_pct.index, dd_pct.values, 0, color="red", alpha=0.45, linewidth=0.0)
    plt.plot(dd_pct.index, dd_pct.values, linewidth=0.8, color="black")
    plt.title("Drawdown (%)")
    plt.xlabel("Time"); plt.ylabel("Drawdown (%)")
    plt.tight_layout()
    plt.savefig(out_dd_pct, dpi=150)
    if show: plt.show()
    plt.close()

def plot_profit_vs_drawdown_dollars(trades: pd.DataFrame, path_png: str, show=True):
    ensure_dir_for(path_png)
    df = trades.copy()
    df["pnl_abs"] = df["pnl"].abs()
    wins, losses = df[df["pnl"] > 0], df[df["pnl"] <= 0]
    plt.figure(figsize=(10,5))
    plt.scatter(wins["mae_dollars"], wins["pnl_abs"], marker="^", s=28, c="green", label="Profitable Trade", edgecolors="none", alpha=0.8)
    plt.scatter(losses["mae_dollars"], losses["pnl_abs"], marker="v", s=28, c="red", label="Losing Trade", edgecolors="none", alpha=0.9)
    plt.xlabel("DrawDown ($)"); plt.ylabel("Profit (Loss) in $")
    plt.title(f"Profit (|P&L|) vs DrawDown ($) — MA(33,98){stop_suffix()}")
    plt.grid(True, linestyle=":", linewidth=0.7); plt.legend(frameon=True).get_frame().set_alpha(0.9)
    plt.gca().set_ylim(bottom=0); plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    if show: plt.show()
    plt.close()

def plot_profit_vs_drawdown_pct(trades: pd.DataFrame, path_png: str, stop_loss_pct: float, show=True):
    ensure_dir_for(path_png)
    df = trades.copy()
    df["ret_pct_abs"] = df["ret_pct"].abs()
    wins, losses = df[df["ret_pct"] > 0], df[df["ret_pct"] <= 0]
    plt.figure(figsize=(10,5))
    plt.scatter(wins["mae_pct"], wins["ret_pct_abs"], marker="^", s=28, c="green", label="Profitable Trade", edgecolors="none", alpha=0.8)
    plt.scatter(losses["mae_pct"], losses["ret_pct_abs"], marker="v", s=28, c="red", label="Losing Trade", edgecolors="none", alpha=0.9)

    # Loss diagonal
    xmax = float(np.nanmax(df["mae_pct"])) if len(df) else 0.0
    ymax = float(np.nanmax(df["ret_pct_abs"])) if len(df) else 0.0
    m = max(xmax, ymax) * 1.05 if (xmax or ymax) else 1.0
    xs = np.linspace(0, m, 200)
    plt.plot(xs, xs, linestyle="-", linewidth=1.4, label="Loss diagonal")

    # Stop marker (only if enabled)
    if USE_STOP_LOSS:
        x = stop_loss_pct * 100.0
        plt.axvline(x, linestyle="--", linewidth=1.2)
        plt.text(x, m*0.9, "Stop loss", rotation=90, ha="right", va="top")

    plt.xlabel("Drawdown in %"); plt.ylabel("Profit (Loss) in %")
    plt.title(f"Maximum Adverse Excursion in % — MA(33,98){stop_suffix()}")
    plt.grid(True, linestyle=":", linewidth=0.7); plt.legend(frameon=True).get_frame().set_alpha(0.9)
    ax = plt.gca(); ax.set_xlim(left=0); ax.set_ylim(bottom=0); plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    if show: plt.show()
    plt.close()

def plot_profit_vs_drawdown_pct_zoom(trades: pd.DataFrame, path_png: str, stop_loss_pct: float, show=True):
    ensure_dir_for(path_png)
    df = trades.copy()
    df["ret_pct_abs"] = df["ret_pct"].abs()
    wins, losses = df[df["ret_pct"] > 0], df[df["ret_pct"] <= 0]
    plt.figure(figsize=(10,5))
    plt.scatter(wins["mae_pct"], wins["ret_pct_abs"], marker="^", s=28, c="green", label="Profitable Trade", edgecolors="none", alpha=0.8)
    plt.scatter(losses["mae_pct"], losses["ret_pct_abs"], marker="v", s=28, c="red", label="Losing Trade", edgecolors="none", alpha=0.9)

    xs = np.linspace(0.0, 0.8, 200)
    plt.plot(xs, xs, linestyle="-", linewidth=1.4, label="Loss diagonal")

    if USE_STOP_LOSS:
        x = stop_loss_pct * 100.0
        if 0.0 <= x <= 0.8:
            plt.axvline(x, linestyle="--", linewidth=1.2)
            y_top = plt.gca().get_ylim()[1]
            plt.text(x, y_top*0.9 if y_top > 0 else 0.5, "Stop loss", rotation=90, ha="right", va="top")

    plt.xlabel("Drawdown in %"); plt.ylabel("Profit (Loss) in %")
    plt.title(f"MAE in % — Zoomed (0.0% to 0.8%) — MA(33,98){stop_suffix()}")
    plt.grid(True, linestyle=":", linewidth=0.7); plt.legend(frameon=True).get_frame().set_alpha(0.9)
    ax = plt.gca(); ax.set_xlim(0.0, 0.8); ax.set_xticks(np.arange(0.0, 0.8 + 1e-9, 0.1)); ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    if show: plt.show()
    plt.close()

# ======== MONTHLY AVERAGE PROFIT (Jan–Dec across years) ========
def monthly_avg_profit(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series([0.0]*12, index=range(1,13), name="avg_pnl")

    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["year"]  = t["exit_time"].dt.year
    t["month"] = t["exit_time"].dt.month

    month_sums = t.groupby(["year", "month"])["pnl"].sum().reset_index()
    avg_by_month = month_sums.groupby("month")["pnl"].mean()
    avg_by_month = avg_by_month.reindex(range(1,13), fill_value=0.0)
    avg_by_month.name = "avg_pnl"
    return avg_by_month

def plot_monthly_avg_profit(avg_series: pd.Series, path_png: str, show=True):
    ensure_dir_for(path_png)
    months = [calendar.month_abbr[m] for m in range(1,13)]
    values = avg_series.values.astype(float)

    plt.figure(figsize=(10,5))
    plt.bar(range(1,13), values, width=0.7, edgecolor="black", color="#7CFC00")
    plt.xticks(range(1,13), months)
    plt.ylabel("Net Profit ($)")
    plt.xlabel("Months")
    plt.title("Average Monthly Profit (across years)")
    plt.grid(True, axis="y", linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    if show: plt.show()
    plt.close()

# ========= MAIN =========
def main():
    df  = load_data(CSV_PATH)
    sig = with_signals(df, FAST, SLOW)
    equity, trades = backtest_with_stop(sig, STOP_LOSS_PCT, use_stop_loss=USE_STOP_LOSS)

    # enrich trades for scatter plots
    if len(trades):
        trades["mae_dollars"] = compute_trade_mae_dollars(sig, trades)
        trades["mae_pct"]     = compute_trade_mae_pct(sig, trades)
        trades["ret_pct"]     = compute_trade_return_pct(trades)
        trades["pnl_abs"]     = trades["pnl"].abs()
        trades["win"]         = trades["pnl"] > 0

    # save trades
    ensure_dir_for(OUT_TRADES)
    trades.to_csv(OUT_TRADES, index=False)

    # charts
    plot_equity_and_drawdown(equity, OUT_EQUITY, OUT_DRAWDOWN_PCT, show=SHOW_PLOTS)
    plot_profit_vs_drawdown_dollars(trades, OUT_SCATTER_DOLLARS, show=SHOW_PLOTS)
    plot_profit_vs_drawdown_pct(trades, OUT_SCATTER_PERCENTAGES, STOP_LOSS_PCT, show=SHOW_PLOTS)
    plot_profit_vs_drawdown_pct_zoom(trades, OUT_SCATTER_PERCENTAGES_ZOOM, STOP_LOSS_PCT, show=SHOW_PLOTS)

    # monthly average profit chart
    avg_series = monthly_avg_profit(trades)
    plot_monthly_avg_profit(avg_series, OUT_MONTHLY_AVG_PROFIT, show=SHOW_PLOTS)

    # performance summary CSV
    perf_table = build_performance_table(trades, equity, n_total_bars=len(sig))
    ensure_dir_for(OUT_PERFORMANCE_CSV)
    perf_table.to_csv(OUT_PERFORMANCE_CSV)

    print("Saved:",
          OUT_TRADES, OUT_EQUITY, OUT_DRAWDOWN_PCT,
          OUT_SCATTER_DOLLARS, OUT_SCATTER_PERCENTAGES, OUT_SCATTER_PERCENTAGES_ZOOM,
          OUT_MONTHLY_AVG_PROFIT, OUT_PERFORMANCE_CSV)

if __name__ == "__main__":
    main()
