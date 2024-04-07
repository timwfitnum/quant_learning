from arch import arch_model
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas_ta
import matplotlib.ticker as mtick

MINUTE_DATA = "5min_data.txt"
DAILY_DATA = "daily_data.txt"


def read_daily_data(file_path):
    """"""
    df = pd.read_csv(file_path)
    df = df.drop("Unnamed: 7", axis=1)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df["log_ret"] = np.log(df["Adj Close"]).diff()
    return df


def read_minute_data(file_path):
    """"""
    df = pd.read_csv(file_path)
    df = df.drop("Unnamed: 6", axis=1)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df["date"] = pd.to_datetime(df.index.date)

    return df


def predict_volatility(df):
    best_model = arch_model(y=df, p=1, q=3, rescale=False).fit(
        update_freq=5, disp="off"
    )

    variance_forecast = best_model.forecast(horizon=1).variance.iloc[-1, 0]
    return variance_forecast


def calculate_prediction_premium(daily_df):
    daily_df["prediction_premium"] = (
        daily_df["predictions"] - daily_df["variance"]
    ) / daily_df["variance"]
    daily_df["premium_std"] = daily_df["prediction_premium"].rolling(180).std()
    daily_df["signal_daily"] = daily_df.apply(
        lambda x: 1
        if (x["prediction_premium"] > x["premium_std"] * 1)
        else -1
        if (x["prediction_premium"] < x["premium_std"] * -1)
        else np.nan,
        axis=1,
    )
    daily_df["signal_daily"] = daily_df["signal_daily"].shift()
    return daily_df


def fit_gaarch_model(daily_df):
    daily_df["variance"] = daily_df["log_ret"].rolling(180).var()
    daily_df = daily_df["2020":]
    daily_df["predictions"] = (
        daily_df["log_ret"].rolling(180).apply(lambda x: predict_volatility(x))
    )
    daily_df = daily_df.dropna()

    return daily_df


def merge_daily_and_intraday(pred_df, minute_df):
    final_df = (
        minute_df.reset_index()
        .merge(pred_df[["signal_daily"]].reset_index(), left_on="date", right_on="Date")
        .set_index("datetime")
    )
    final_df = final_df.drop(["date", "Date"], axis=1)
    return final_df


def calc_indicators(final_df):
    final_df["rsi"] = pandas_ta.rsi(close=final_df["Close"], lenght=20)
    final_df["lband"] = (
        x := pandas_ta.bbands(close=final_df["Close"], length=20)
    ).iloc[:, 0]
    final_df["uband"] = x.iloc[:, 2]
    final_df["signal_intraday"] = final_df.apply(
        lambda x: 1
        if (x["rsi"] > 70 & (x["Close"] > x["uband"]))
        else -1
        if (x["rsi"] < 30 and (x["Close"] < x["lband"]))
        else np.nan,
        axis=1,
    )
    return final_df


def generate_entries(df):
    df["return_sign"] = df.apply(
        lambda x: -1
        if (x["signal_daily"] == 1) & (x["signal_intraday"] == 1)
        else 1
        if (x["signal_daily"] == -1) & (x["signal_intraday"] == -1)
        else np.nan,
        axis=1,
    )
    df["return_sign"] = df.groupby(pd.Grouper(freq="D"))["return_sign"].transform(
        lambda x: x.ffill()
    )
    df["return"] = df["Close"].pct_change()
    df["forward_return"] = df["return"].shift(-1)
    df["Strategy Return"] = df["forward_return"] * df["return_sign"]
    daily_return = df.groupby(pd.Grouper(freq="D"))[["Strategy Return"]].sum()

    return daily_return


def visualise(df):
    strategy_cum_return = np.exp(np.log1p(df).cumsum()).sub(1)
    strategy_cum_return.plot(figsize=(16, 6))
    plt.title("Intraday Strategy Returns")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    daily_df = read_daily_data(DAILY_DATA)
    minute_df = read_minute_data(MINUTE_DATA)
    daily_df = fit_gaarch_model(daily_df)
    pred_df = calculate_prediction_premium(daily_df)
    final_df = merge_daily_and_intraday(pred_df, minute_df)
    indic_df = calc_indicators(final_df)
    entries = generate_entries(indic_df)
    visualise(entries)
