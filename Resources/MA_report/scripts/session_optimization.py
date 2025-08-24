import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= CONFIG =========
CSV_PATH = "../candlesticks/EURUSD_30m.csv"

BEST_FAST = 33
BEST_SLOW = 98

INITIAL_CAPITAL   = 100_000.0
RISK_DOLLARS      = 1_000.0
RISK_MOVE_PCT     = 0.01
COMMISSION_PER_RT = 25.0
SLIPPAGE_PER_RT   = 5.0
COST_PER_RT       = COMMISSION_PER_RT + SLIPPAGE_PER_RT

WINDOW_HOURS = 4
STEP_MINUTES = 30

REQUIRE_MIN_TRADES = 1
SHOW_PLOTS = True

OUT_RESULTS_CSV = "../metrics/ma_session_sweep.csv"
OUT_PNG         = "../graphics/ma_session_curve.png"
OUT_TRADES      = "../metrics/trades_session.csv"
OUT_EQUITY      = "../graphics/equity_curve_session.png"
OUT_DD_PCT      = "../graphics/drawdown_percent_session.png"
OUT_SUMMARY     = "../metrics/ma_session_summary.csv"

# ========= IO / LOADING =========
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

    o = _match(df, ["open"]); h = _match(df, ["high"]); l = _match(df, ["low"])
    c = _match(df, ["close","adj close"])
    if c is None: raise ValueError("CSV must contain a 'close' column.")

    df = df.rename(columns={dt: "datetime"})
    df["open"]  = df[o] if o else df[c]
    df["high"]  = df[h] if h else df[c]
    df["low"]   = df[l] if l else df[c]
    df["close"] = df[c]
    return df[["datetime","open","high","low","close"]]

# ========= SIGNALS =========
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

# ========= TIME WINDOW HELPERS =========
def minutes_since_midnight(ts: pd.Timestamp) -> int:
    t = ts.time()
    return t.hour*60 + t.minute

def in_window(mins: int, start_min: int, dur_min: int) -> bool:
    end = (start_min + dur_min) % 1440
    if dur_min >= 1440:
        return True
    if start_min <= end:
        return start_min <= mins < end
    # wrap over midnight
    return mins >= start_min or mins < end

# ========= BACKTEST (entries/reversals only in window) =========
TRADE_COLS = [
    "entry_time","exit_time","side","units",
    "entry_price","exit_price","bars_held",
    "gross_pnl","costs_rt","pnl"
]

def backtest_window(df_sig: pd.DataFrame, start_minute: int, window_hours: int):
    cash = INITIAL_CAPITAL
    direction = 0
    units = 0.0
    entry_price = np.nan
    entry_idx = None

    equity_times, equity_vals = [], []
    trades_list = []

    dur_min = int(window_hours * 60)

    for i in range(1, len(df_sig)-1):
        row = df_sig.iloc[i]
        nxt = df_sig.iloc[i+1]

        # mark-to-market
        mtm = cash + (units * (row["close"] - entry_price) if direction != 0 else 0.0)
        equity_times.append(row["datetime"])
        equity_vals.append(mtm)

        if row["signal"] != 0:
            mins = minutes_since_midnight(row["datetime"])
            if in_window(mins, start_minute, dur_min):
                # exit existing
                if direction != 0:
                    exit_price = nxt["open"]
                    gross_pnl = units * (exit_price - entry_price)
                    net_pnl   = gross_pnl - COST_PER_RT
                    cash += net_pnl
                    trades_list.append({
                        "entry_time": df_sig.iloc[entry_idx]["datetime"],
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

                # enter new
                direction = int(row["signal"])
                assumed_move = max(RISK_MOVE_PCT * row["close"], 1e-12)
                size_units = RISK_DOLLARS / assumed_move
                units = direction * size_units
                entry_price = nxt["open"]
                entry_idx = i+1

    # liquidate on last bar close
    last = df_sig.iloc[-1]
    if direction != 0:
        exit_price = last["close"]
        gross_pnl = units * (exit_price - entry_price)
        net_pnl   = gross_pnl - COST_PER_RT
        cash += net_pnl
        trades_list.append({
            "entry_time": df_sig.iloc[entry_idx]["datetime"],
            "exit_time":  last["datetime"],
            "side": "LONG" if direction==1 else "SHORT",
            "units": units,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bars_held": len(df_sig)-1 - entry_idx + 1,
            "gross_pnl": gross_pnl,
            "costs_rt": COST_PER_RT,
            "pnl": net_pnl,
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

def _streaks(bools):
    best = cur = 0
    for b in bools:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best

def _block(tr_df: pd.DataFrame, equity: pd.Series):
    pnl = tr_df["pnl"] if "pnl" in tr_df.columns else pd.Series(dtype=float)
    wins = pnl[pnl > 0]; losses = pnl[pnl <= 0]
    peak = equity.cummax()
    dd_amt = (equity - peak).min() if len(equity) else 0.0
    mdd_amt = -float(dd_amt) if dd_amt is not None else 0.0
    mdd_date = (equity - peak).idxmin() if len(equity) else pd.NaT
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
        "Max. Drawdown (Intraday Peak to Valley)": mdd_amt,
        "Date of Max. Drawdown": mdd_date.strftime("%d-%b-%y") if pd.notna(mdd_date) else "",
    }

def build_table(trades: pd.DataFrame, equity: pd.Series):
    all_blk   = _block(trades, equity)
    long_blk  = _block(trades[trades["side"]=="LONG"], equity)
    short_blk = _block(trades[trades["side"]=="SHORT"], equity)
    rows = list(all_blk.keys())
    out = pd.DataFrame(
        {"All Trades":[all_blk[r] for r in rows],
         "Long Trades":[long_blk[r] for r in rows],
         "Short Trades":[short_blk[r] for r in rows]},
        index=rows,
    )
    return out

# ========= ROBUST PICK =========
def pick_plateau_mid(series: pd.Series, frac_of_max: float = 0.90):
    """Return the index label at the midpoint of the widest contiguous region with y >= frac*max."""
    if series.empty:
        return None
    vmax = series.max()
    thresh = frac_of_max * vmax
    mask = (series >= thresh).astype(int).values
    best_len, best_start = 0, None
    cur_len, cur_start = 0, None
    for i, v in enumerate(mask):
        if v == 1:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len, cur_start = 0, None
    if best_len == 0:
        return series.idxmax()
    mid = best_start + (best_len - 1)//2
    return series.index[mid]

# ========= MAIN =========
def main():
    df = load_data(CSV_PATH)
    df_sig = with_signals(df, BEST_FAST, BEST_SLOW)

    starts = list(range(0, 24*60, STEP_MINUTES))  # 00:00, 00:30, ..., 23:30
    results = []

    for start_min in starts:
        eq, tr = backtest_window(df_sig, start_min, WINDOW_HOURS)
        end_eq = float(eq.iloc[-1]) if len(eq) else INITIAL_CAPITAL
        netp   = end_eq - INITIAL_CAPITAL
        n_tr   = len(tr)

        start_h, start_m = divmod(start_min, 60)
        end_min = (start_min + WINDOW_HOURS*60) % 1440
        end_h, end_m = divmod(end_min, 60)
        label = f"{start_h:02d}:{start_m:02d}-{end_h:02d}:{end_m:02d}"

        results.append((start_min, end_min, label, end_eq, netp, n_tr))

    res = pd.DataFrame(results, columns=["start_min","end_min","window","end_equity","net_profit","n_trades"])
    res.to_csv(OUT_RESULTS_CSV, index=False)

    # Sort by chronological start and prep plotting vectors
    res = res.sort_values("start_min").reset_index(drop=True)
    x = np.arange(len(res))
    y = res["net_profit"].values
    windows = res["window"].values

    # Robust pick (midpoint of widest >=90% plateau)
    series = pd.Series(y, index=windows)
    robust_window = pick_plateau_mid(series, frac_of_max=0.90)
    robust_row = res[res["window"] == robust_window].iloc[0]
    robust_idx = int(res.index[res["window"] == robust_window][0])

    # === Plot like the book: area fill + markers + vertical line at robust midpoint ===
    plt.figure(figsize=(12,4.5))
    # area fill (shade above zero for positive)
    plt.fill_between(x, np.maximum(y, 0), 0, alpha=0.35)
    plt.plot(x, y, marker="o", linewidth=1.5, markersize=3)

    # vertical line at robust midpoint
    plt.axvline(robust_idx, color="k", linewidth=1.25)

    # x-ticks every hour (i.e., every 2 points since we step 30min)
    tick_every = 2
    tick_idx = list(range(0, len(x), tick_every))
    tick_labels = [windows[i].split("-")[0] for i in tick_idx]
    plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")

    plt.title(f"Net Profit vs Start Time — 4h Window (FAST={BEST_FAST}, SLOW={BEST_SLOW}, $30 RT)")
    plt.xlabel("Start Time (local in file)")
    plt.ylabel("Net Profit ($)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    if SHOW_PLOTS: plt.show()
    plt.close()

    # Backtest using the ROBUST start (not the absolute max), per process on p.20
    eq, tr = backtest_window(df_sig, int(robust_row.start_min), WINDOW_HOURS)
    tr.to_csv(OUT_TRADES, index=False)

    # Equity
    plt.figure(figsize=(10,4))
    plt.plot(eq.index, eq.values)
    plt.title(f"Equity Curve — Window {robust_row.window} (FAST={BEST_FAST}, SLOW={BEST_SLOW})")
    plt.xlabel("Time"); plt.ylabel("Equity ($)")
    plt.tight_layout(); plt.savefig(OUT_EQUITY, dpi=150)
    if SHOW_PLOTS: plt.show()
    plt.close()

    # Drawdown %
    dd_pct = drawdown_percent_series(eq) * 100.0
    plt.figure(figsize=(10,3.6))
    plt.fill_between(dd_pct.index, dd_pct.values, 0)
    plt.title("Drawdown (%)")
    plt.xlabel("Time"); plt.ylabel("Drawdown (%)")
    plt.tight_layout(); plt.savefig(OUT_DD_PCT, dpi=150)
    if SHOW_PLOTS: plt.show()
    plt.close()

    # Summary table
    summary = build_table(tr, eq)
    summary.to_csv(OUT_SUMMARY)

    # Console hints
    best_idx = int(res["net_profit"].idxmax())
    best_row = res.loc[best_idx]
    print(f"Best start (max Net): {best_row.window} | Trades={int(best_row.n_trades)} | Net=${best_row.net_profit:,.0f}")
    print(f"Robust midpoint (>=90% of max): {robust_row.window} | Trades={int(robust_row.n_trades)} | Net=${robust_row.net_profit:,.0f}")
    print(f"Saved: {OUT_RESULTS_CSV}, {OUT_PNG}, {OUT_TRADES}, {OUT_EQUITY}, {OUT_DD_PCT}, {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
