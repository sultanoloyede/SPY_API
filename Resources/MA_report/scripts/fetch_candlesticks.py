import re, io, os
import requests
import pandas as pd

# ================== CONFIG ==================
SYMBOL     = "EUR/USD"          # e.g., "EUR/USD"
TIMEFRAME  = "30m"              # e.g., "30m", "8h", "2D", "W", "M"
START      = "2019-01-01"
END        = "2025-01-01"
OUTDIR     = "."                # where to save the CSV
# ============================================

def fxcm_symbol(sym: str) -> str:
    return sym.replace("/", "").upper()

def parse_timeframe(tf: str) -> tuple[str, str]:
    """
    Returns (pandas_freq, base_periodicity) where base_periodicity in {'m1','H1','D1'}.
    """
    s = tf.strip().lower()

    # common aliases
    alias = {
        "1m":"1min","m1":"1min","min":"1min",
        "1h":"1h","h1":"1h","hour":"1h",
        "1d":"1d","d1":"1d","day":"1d",
        "w":"w","1w":"w",
        "m":"m","1mo":"m","1mth":"m",
    }
    if s in alias: s = alias[s]

    # accept patterns like '30m','90min','2h','8H','2D','W','M'
    m = re.fullmatch(r"(\d+)\s*(min|m|h|d|w|mo|mth)?", s, flags=re.I)
    if m:
        n = int(m.group(1))
        unit = (m.group(2) or "min").lower()
        unit = {"m":"min","h":"h","d":"d","mo":"m","mth":"m"}.get(unit, unit)
        pandas_freq = f"{n}{ {'min':'T','h':'H','d':'D','w':'W','m':'M'}[unit] }"
    else:
        # bare 'w' or 'm' etc.
        if s in {"w","week"}:
            pandas_freq = "W"
        elif s in {"m","month"}:
            pandas_freq = "M"
        elif s in {"d","day"}:
            pandas_freq = "1D"
        else:
            raise ValueError(f"Unrecognized timeframe '{tf}'")

    # Decide best base (m1/H1/D1)
    offset = pd.tseries.frequencies.to_offset(pandas_freq)
    minutes = int(offset.delta.total_seconds() // 60) if offset.delta is not None else None

    if minutes is None:
        # calendar-anchored like W/M -> use D1
        base = "D1"
    elif minutes < 60:
        base = "m1"
    elif minutes < 24*60:
        base = "H1" if (minutes % 60 == 0) else "m1"
    else:
        base = "D1"

    return pandas_freq, base

def _pick_ohlc(df: pd.DataFrame):
    cols_lc = {c.lower(): c for c in df.columns}
    def pick(prefix):
        need = [prefix+"open", prefix+"high", prefix+"low", prefix+"close"]
        if all(k in cols_lc for k in need):
            return (
                df[cols_lc[need[0]]].astype(float).rename("open"),
                df[cols_lc[need[1]]].astype(float).rename("high"),
                df[cols_lc[need[2]]].astype(float).rename("low"),
                df[cols_lc[need[3]]].astype(float).rename("close"),
            )
        return None
    ohlc = pick("bid") or pick("ask")
    if ohlc is None and all(k in cols_lc for k in ["bidopen","askopen","bidhigh","askhigh","bidlow","asklow","bidclose","askclose"]):
        open_  = (df[cols_lc["bidopen"]]  + df[cols_lc["askopen"]])  / 2
        high_  = (df[cols_lc["bidhigh"]]  + df[cols_lc["askhigh"]])  / 2
        low_   = (df[cols_lc["bidlow"]]   + df[cols_lc["asklow"]])   / 2
        close_ = (df[cols_lc["bidclose"]] + df[cols_lc["askclose"]]) / 2
        ohlc = (open_.astype(float).rename("open"),
                high_.astype(float).rename("high"),
                low_.astype(float).rename("low"),
                close_.astype(float).rename("close"))
    if ohlc is None:
        generic = ["open","high","low","close"]
        if all(k in cols_lc for k in generic):
            ohlc = tuple(df[cols_lc[k]].astype(float).rename(k) for k in generic)
        else:
            raise ValueError("Could not locate OHLC columns.")
    return ohlc

def _read_gz_csv(url: str, session: requests.Session) -> pd.DataFrame | None:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = session.get(url, headers=headers, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content), compression="gzip")

def load_fxcm_base(symbol: str, start: str, end: str, base: str) -> pd.DataFrame:
    """
    base in {'m1','H1','D1'}
    """
    sdt = pd.to_datetime(start, utc=True)
    edt = pd.to_datetime(end, utc=True)
    session = requests.Session()
    frames = []

    if base in ("m1","H1"):
        # weekly files over the day range
        days = pd.date_range(sdt.normalize(), edt.normalize(), freq="D", tz="UTC")
        iso = days.isocalendar()
        year_week = sorted(set(zip(iso["year"].tolist(), iso["week"].tolist())))
        for y, w in year_week:
            url = f"https://candledata.fxcorporate.com/{base}/{symbol}/{y}/{int(w)}.csv.gz"
            tmp = _read_gz_csv(url, session)
            if tmp is not None:
                frames.append(tmp)
    elif base == "D1":
        years = range(sdt.year, edt.year + 1)
        for y in years:
            url = f"https://candledata.fxcorporate.com/D1/{symbol}/{y}.csv.gz"
            tmp = _read_gz_csv(url, session)
            if tmp is not None:
                frames.append(tmp)
    else:
        raise ValueError("Invalid base")

    if not frames:
        raise RuntimeError("No FXCM data loaded—check symbol/date span.")

    raw = pd.concat(frames, ignore_index=True)
    # index by timestamp
    cols_lc = {c.lower(): c for c in raw.columns}
    ts_col = cols_lc.get("datetime") or cols_lc.get("date")
    if ts_col is None:
        raise ValueError("No Date/DateTime column found.")
    ts = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
    df = raw.set_index(ts).sort_index()
    o,h,l,c = _pick_ohlc(df)

    out = pd.DataFrame({"open":o,"high":h,"low":l,"close":c})
    out = out[(out.index >= sdt) & (out.index <= edt)]
    out.index.name = "timestamp"
    return out

def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    rs = df.resample(freq, label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}
    ).dropna()
    rs.index.name = "timestamp"
    return rs

def main(symbol_in: str, tf_in: str, start: str, end: str, outdir: str):
    fxcm_sym = fxcm_symbol(symbol_in)
    freq, base = parse_timeframe(tf_in)
    base_df = load_fxcm_base(fxcm_sym, start, end, base)

    # If target equals base (within reason), just use base_df
    same = ((base == "m1" and freq in {"1T","1min"})
            or (base == "H1" and freq in {"1H"})
            or (base == "D1" and freq in {"1D"}))

    df = base_df if same else resample_ohlc(base_df, freq)

    # trim to exact [start, end]
    sdt = pd.to_datetime(start, utc=True); edt = pd.to_datetime(end, utc=True)
    df = df[(df.index >= sdt) & (df.index <= edt)]

    # save
    path = "../candlesticks/EURUSD_30m.csv"
    df.to_csv(path)
    print(f"Saved {len(df):,} rows to {path}")

if __name__ == "__main__":
    main(SYMBOL, TIMEFRAME, START, END, OUTDIR)
