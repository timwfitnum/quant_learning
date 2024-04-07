import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
import matplotlib.ticker as mtick

data_folder = "twitter_data.txt"


def get_twitter_data() -> pd.DataFrame:
    """"""
    sentiment_df = pd.read_csv(data_folder)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    sentiment_df = sentiment_df.set_index(["date", "symbol"])

    return sentiment_df


def create_indicators(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """"""
    sentiment_df["engagement_ratio"] = (
        sentiment_df["twitterComments"] / sentiment_df["twitterLikes"]
    )
    sentiment_df = sentiment_df[
        (sentiment_df["twitterLikes"] > 20) & (sentiment_df["twitterComments"] > 10)
    ]
    agg_df = (
        sentiment_df.reset_index("symbol")
        .groupby([pd.Grouper(freq="M"), "symbol"])[["engagement_ratio"]]
        .mean()
    )

    return agg_df


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """"""
    data["rank"] = data.groupby(level=0)["engagement_ratio"].transform(
        lambda x: x.rank(ascending=False)
    )
    t_data = data[data["rank"] < 6].copy()
    t_data = t_data.reset_index(level=1)
    t_data.index = t_data.index + pd.DateOffset(days=1)
    t_data = t_data.reset_index().set_index(["date", "symbol"])
    return t_data


def extract_data(data: pd.DataFrame) -> dict:
    """"""
    dates = data.index.get_level_values("date").unique().tolist()

    fixed_dates = {
        d.strftime("%Y-%m-%d"): data.xs(d, level=0).index.tolist() for d in dates
    }
    return fixed_dates


def get_stock_data(stock_list, fixed_dates) -> pd.DataFrame:
    """"""
    start = (x := list(fixed_dates.keys()))[0]
    end = x[-1]
    df = yf.download(tickers=stock_list, start=start, end=end)
    return df


def get_nasdaq_data():
    """"""
    nas_df = yf.download(tickers="QQQ", start="2021-01-01", end="2023-03-01")
    qqq_ret = np.log(nas_df["Adj Close"]).diff().to_frame("Nasdaq Return")
    return qqq_ret


def get_portfolio_returns(stock_data, t_data, dates) -> pd.DataFrame:
    """"""
    returns_df = np.log(stock_data["Adj Close"]).diff().dropna()
    portfolio_df = pd.DataFrame()
    for start in dates.keys():
        end_date = (pd.to_datetime(start) + pd.offsets.MonthEnd()).strftime("%Y-%m-%d")
        cols = dates[start]
        try:
            cols.remove("ATVI")
        except ValueError:
            pass
        temp_df = (
            returns_df[start:end_date][cols].mean(axis=1).to_frame("portfolio_return")
        )
        portfolio_df = pd.concat([portfolio_df, temp_df])
    return portfolio_df


def visualise_data(portfolio_df: pd.DataFrame):
    """"""
    port_cum_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)
    port_cum_return.plot(figsize=(16, 6))

    plt.title("Twitter Engagement Ratio Strategy Return Over Time")

    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Return")
    plt.show()


if __name__ == "__main__":
    sent_df: pd.DataFrame = get_twitter_data()
    agg_df = create_indicators(sent_df)
    t_data = transform_data(agg_df)
    dates = extract_data(t_data)
    stock_list = t_data.index.get_level_values("symbol").unique().tolist()
    try:
        stock_list.remove("ATVI")
    except ValueError:
        pass
    stock_data = get_stock_data(stock_list, dates)
    port_returns = get_portfolio_returns(stock_data, t_data, dates)
    nas_df = get_nasdaq_data()
    portfolio_df = port_returns.merge(nas_df, left_index=True, right_index=True)
    visualise_data(portfolio_df)
