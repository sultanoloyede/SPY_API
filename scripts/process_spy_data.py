import pandas as pd
import os

def above_below_prev_day_close(df):
    # 1. Sort by datetime to ensure “first bar” is really the earliest
    df = df.sort_values('datetime')

    # 2. Create a date-only column for grouping
    df['date'] = df['datetime'].dt.date

    # 3. Compute the first-bar close for each date
    first_close = (
        df
        .groupby('date', sort=True)['close']
        .first()           # this is the close of the first 15m bar each day
    )

    # 4. Shift that Series down by one day
    prev_first_close = first_close.shift(1)

    # 5. Map that back onto every row via the date column
    df['prev_day_first_close'] = df['date'].map(prev_first_close)

    # 6. Compare each bar’s close to the previous day’s first-bar close
    df['close_v_ystd_fst_close'] = (df['close'] > df['prev_day_first_close']).astype(int)

    # 7. Drop helper columns
    df = df.drop(columns=['date', 'prev_day_first_close'])

    return df

def above_below_curr_day_close(df):
    # 1. Sort by datetime to ensure “first bar” is really the earliest
    df = df.sort_values('datetime')

    # 2. Create a date-only column for grouping
    df['date'] = df['datetime'].dt.date

    # 3. Compute the first-bar close for each date
    first_close = (
        df
        .groupby('date', sort=True)['close']
        .first()           # this is the close of the first 15m bar each day
    )

    # 4. Map that back onto every row via the date column
    df['curr_day_first_close'] = df['date'].map(first_close)

    # 5. Compare each bar’s close to the current day’s first-bar close
    df['close_v_fst_close'] = (df['close'] > df['curr_day_first_close']).astype(int)

    # 6. Drop helper columns
    df = df.drop(columns=['date', 'curr_day_first_close'])

    return df

def ema200(df):
    # 1) Sort by datetime columns
    df = df.sort_values('datetime')
    # 2) Calculate 200 ema
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    # 3) Flag if price/candlestick closes above or below
    df['above_200_ema'] = (df['close'] > df['ema_200']).astype(int)
    # 4) Drop helper columns
    df = df.drop(columns=['ema_200'])

    return df
    
def positive_big_move(df):
    # 1) Sort by datetime columns
    df = df.sort_values('datetime')
    # 2) Create a column to calculate growth/decrement of spy value
    df['move_percentage'] = ((df['close'] / df['open']) - 1) * 100
    # 3) Check if its a big grower -> value equal to/greater 0.25
    df['0.25_growth'] = (df['move_percentage'] >= 0.25).astype(int)

    return df

def big_move_counter(df):
    # 1) Sort by datetime columns
    df = df.sort_values('datetime')
    # 2) Check if we have a big drop -> value equal to/less than 0.25
    df['0.25_decrement'] = (df['move_percentage'] <= -0.25).astype(int)
    # 3) Create a date column which will help with grouping later
    df['date'] = df['datetime'].dt.date
    # 4) Create a big move column that will track if a big move happened in that candlestick
    df['big_move'] = df['0.25_growth'] - df['0.25_decrement']
    # 5) Cummulative sum big moves that happened up until the current candlestick within the day
    df['big_move_counter'] = (
        df.groupby('date')['big_move']
        .cumsum()
    )
    # 6) Drop helper columns
    df = df.drop(columns=['date', 'big_move'])

    return df

def rsi(df):
    # 1) Sort by datetime columns
    df = df.sort_values('datetime')
    # 2) 
    delta = df['close'].diff()
    # 3)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    avg_gain = avg_gain.combine_first(
        gain.ewm(alpha=1/14, adjust=False).mean()
    )
    avg_loss = avg_loss.combine_first(
        loss.ewm(alpha=1/14, adjust=False).mean()
    )
    # 4)
    rs = avg_gain / avg_loss
    # 5) 
    df['RSI_14'] = 100 - (100 / (1 + rs)) 
    # 6) 
    df['RSI_above_60'] = (df['RSI_14'] >= 60).astype(int)
    # 7)
    df = df.drop(columns=['RSI_14'])

    return df

df = pd.read_csv("../data/spy_15min.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

df = above_below_prev_day_close(df)
df = above_below_curr_day_close(df)
df = ema200(df)
df = positive_big_move(df)
df = big_move_counter(df)
df = rsi(df)

# 9. Save the result
df.to_csv("../data/processed_spy_15min.csv", index=True)
print(df.head())



