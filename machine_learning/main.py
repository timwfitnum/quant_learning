from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web_reader
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import pandas_ta
from sklearn.cluster import KMeans
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from collections import OrderedDict
from 
from multipledispatch import dispatch

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _get_data(symbols, start_date, end_date) -> pd.DataFrame:
    # data = yf.download(tickers=symbols, start=start_date, end=end_date).stack()
    # data.to_csv("data.txt")
    return pd.read_csv("data.txt", parse_dates=["Date"]).set_index(["Date", "Ticker"])


def fetch_and_clean_data() -> pd.DataFrame:
    sp500 = pd.read_html(WIKI_URL)[0]
    sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-")
    symbols = sp500["Symbol"].unique().tolist()  # Suvivorship bias present.

    end_date = "2024-02-01"
    start_date = pd.to_datetime(end_date) - pd.DateOffset(365 * 8)

    df = _get_data(tuple(symbols), start_date, end_date)
    df.index.names = ["date", "ticker"]
    df.columns = df.columns.str.lower()
    return df


def create_german_klass_vol(data: pd.DataFrame):
    return ((np.log(data["high"]) - np.log(data["low"])) ** 2) / 2 - (
        2 * np.log(2) - 1
    ) * ((np.log(data["adj close"]) - np.log(data["open"])) ** 2)


def create_rsi(data: pd.DataFrame):
    return data.groupby(level=1)["adj close"].transform(
        lambda x: pandas_ta.rsi(close=x, length=20)
    )


def create_bollingers(data: pd.DataFrame):
    data["bb_low"] = data.groupby(level=1)["adj close"].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0]
    )
    data["bb_med"] = data.groupby(level=1)["adj close"].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1]
    )
    data["bb_high"] = data.groupby(level=1)["adj close"].transform(
        lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2]
    )
    return data


def create_atr(stock_data: pd.DataFrame):
    atr = pandas_ta.atr(
        high=stock_data["high"],
        low=stock_data["low"],
        close=stock_data["close"],
        length=14,
    )
    return atr.sub(atr.mean()).div(atr.std())


def create_macd(close: pd.DataFrame):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())


def create_dollor_vol(data: pd.DataFrame):
    return (data["adj close"] * data["volume"]) / 1e6


def create_indicators(data: pd.DataFrame):
    """"""
    data["gkvol"] = create_german_klass_vol(data)
    data["rsi"] = create_rsi(data)
    # data.xs("AAPL", level=1)["rsi"].plot(color="r")
    data = create_bollingers(data)
    data["atr"] = data.groupby(level=1, group_keys=False).apply(create_atr)
    data["macd"] = data.groupby(level=1, group_keys=False)["adj close"].apply(
        create_macd
    )
    data["dollar_vol"] = create_dollor_vol(data)
    return data


def aggregate_data(ind_data: pd.DataFrame):
    """Aggregate to monthly level and filter top 150 most liquid stocks"""
    indicator_cols = [
        c
        for c in ind_data.columns.unique(0)
        if c not in ["dollar_vol", "vol", "open", "high", "low", "open"]
    ]
    data = ind_data.unstack()[indicator_cols].resample("M").last().stack("ticker")
    vol = (
        ind_data.unstack("ticker")
        .loc[:, "dollar_vol"]
        .resample("M")
        .mean()
        .stack("ticker")
        .to_frame("dollar_vol")
    )
    agg_data = pd.concat([data, vol], axis=1).dropna()
    agg_data["dollar_vol"] = (
        agg_data.loc[:, "dollar_vol"]
        .unstack("ticker")
        .rolling(5 * 12, min_periods=12)
        .mean()
        .stack()
    )
    agg_data["dollar_vol_rank"] = agg_data.groupby("date")["dollar_vol"].rank(
        ascending=False
    )
    agg_data = agg_data[agg_data["dollar_vol_rank"] < 150].drop(
        ["dollar_vol", "dollar_vol_rank"], axis=1
    )
    return agg_data


def calcualte_monthly_returns(agg_data: pd.DataFrame):
    """Calculate Monthly Returns for different time horizons.
    Capture Time series dynamics that reflect momentum patterns etc.
    Compute Historical returns using .pct_change over various time periods.
    """

    outlier_cutoff = 0.005
    month_lags = [1, 2, 3, 6, 9, 12]
    for lag in month_lags:
        agg_data[f"return_{lag}m"] = (
            agg_data["adj close"]
            .pct_change(lag)
            .pipe(
                lambda x: x.clip(
                    lower=x.quantile(outlier_cutoff),
                    upper=x.quantile(1 - outlier_cutoff),
                )
            )
            .add(1)
            .pow(1 / lag)
            .sub(1)
        )

    return agg_data


def fama_french_factors():
    """
    Why?

    What is it?
    """
    factor_data = web_reader.DataReader(
        "F-F_Research_Data_5_Factors_2x3", "famafrench", start="2010"
    )[0].drop("RF", axis=1)
    factor_data.index = factor_data.index.to_timestamp()
    factor_data = factor_data.resample("M").last().div(100)
    factor_data.index.name = "date"
    return factor_data


def get_most_liquid(return_data: pd.DataFrame):
    factor_data = fama_french_factors()
    joined_data = factor_data.join(return_data["return_1m"]).sort_index()
    observations = joined_data.groupby(level=1).size()
    valid_stocks = observations[observations >= 10]
    liquid_stocks = joined_data[
        joined_data.index.get_level_values("ticker").isin(valid_stocks.index)
    ]
    return liquid_stocks


def agg_missing_data(data: pd.DataFrame):
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    data.loc[:, cols] = data.groupby("ticker", group_keys=False)[cols].apply(
        lambda x: x.fillna(x.mean())
    )
    return data.dropna()


def get_feature_data():
    data = fetch_and_clean_data()
    ind_data: pd.DataFrame = create_indicators(data)
    agg_data: pd.DataFrame = aggregate_data(ind_data)
    return_data = (
        agg_data.groupby(level=1, group_keys=False)
        .apply(calcualte_monthly_returns)
        .dropna()
    )
    liquid_stocks: pd.DataFrame = get_most_liquid(return_data)
    beta_data = (
        liquid_stocks.groupby(level=1, group_keys=False)
        .apply(
            lambda x: RollingOLS(
                endog=x["return_1m"],
                exog=sm.add_constant(x.drop("return_1m", axis=1)),
                window=min(24, x.shape[0]),
                min_nobs=len(x.columns) + 1,
            )
            .fit()
            .params
        )
        .drop("const", axis=1)
    ).shift()
    meshed_data: pd.DataFrame = return_data.join(beta_data.groupby("ticker").shift())
    ave_data: pd.DataFrame = agg_missing_data(meshed_data)
    feature_data: pd.DataFrame = ave_data.drop(["adj close", "close", "volume"], axis=1)
    return feature_data


def plot_clusters(data: pd.DataFrame):
    cluster_0 = data[data["cluster"] == 0]
    cluster_1 = data[data["cluster"] == 1]
    cluster_2 = data[data["cluster"] == 2]
    cluster_3 = data[data["cluster"] == 3]

    plt.scatter(cluster_0["atr"], cluster_0["rsi"], color="red", label="cluster_0")
    plt.scatter(cluster_1["atr"], cluster_1["rsi"], color="green", label="cluster_1")
    plt.scatter(cluster_2["atr"], cluster_2["rsi"], color="blue", label="cluster_2")
    plt.scatter(cluster_3["atr"], cluster_3["rsi"], color="black", label="cluster_3")

    plt.legend()
    plt.show()


def plot_data(feature_data: pd.DataFrame):
    plt.style.use("ggplot")
    for i in feature_data.index.get_level_values("date").unique().tolist():
        g = feature_data.xs(i, level=0)
        plt.title(f"Date {i}")
        # plot_clusters(g)
        # breakpoint()


def fit_k_means(feature_data: pd.DataFrame):
    """"""
    target_rsi_values = [30, 45, 55, 70]
    initial_centroids = np.zeros((len(target_rsi_values), 18))
    initial_centroids[:, feature_data.columns.get_loc("rsi")] = target_rsi_values

    def _get_clusters(df: pd.DataFrame):
        df["cluster"] = (
            KMeans(n_clusters=4, random_state=0, init=initial_centroids).fit(df).labels_
        )
        return df

    feature_data = (
        feature_data.dropna().groupby("date", group_keys=False).apply(_get_clusters)
    )
    # plot_data(feature_data)
    return feature_data


def get_dates_and_tickers(cluster_data: pd.DataFrame, cluster: int = 3):
    high_rsi = cluster_data[cluster_data["cluster"] == cluster].copy()
    filtered_data = high_rsi.reset_index(level=1)
    filtered_data.index = filtered_data.index + pd.DateOffset(1)

    filtered_data = filtered_data.reset_index().set_index(["date", "ticker"])
    dates: list[pd.DatetimeIndex] = (
        filtered_data.index.get_level_values("date").unique().tolist()
    )
    fixed_dates = {
        d.strftime("%Y-%m-%d"): filtered_data.xs(d, level=0).index.tolist()
        for d in dates
    }
    return fixed_dates


def optimize_weights(prices, lower_bound=0) -> OrderedDict:
    """"""
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    covariance = risk_models.sample_cov(prices=prices, frequency=252)

    ef = EfficientFrontier(
        expected_returns=returns,
        cov_matrix=covariance,
        weight_bounds=(lower_bound, 0.1),
        solver="SCS",
    )

    weights = ef.max_sharpe()
    return ef.clean_weights()


def get_daily_prices(df: pd.DataFrame):
    """"""
    stocks = df.index.get_level_values("ticker").unique().tolist()
    dates = df.index.get_level_values("date").unique()
    price_df = yf.download(
        tickers=stocks, start=dates[0] - pd.DateOffset(months=12), end=dates[-1]
    )
    return price_df


def monthly_optimise(new_df, dates) -> pd.DataFrame:
    returns_df = np.log(new_df["Adj Close"]).diff()
    portfolio_df = pd.DataFrame()
    for start in dates.keys():
        try:
            end_date = (pd.to_datetime(start) + pd.offsets.MonthEnd(0)).strftime(
                "%Y-%m-%d"
            )
            cols = dates[start]
            optimize_start_date = pd.to_datetime(start) - pd.DateOffset(months=12)
            optimize_end_date = pd.to_datetime(start) - pd.DateOffset(days=1)
            # breakpoint()
            optimise_df = new_df[optimize_start_date:optimize_end_date]["Adj Close"][
                cols
            ]
            try:
                weights = pd.DataFrame(
                    optimize_weights(
                        optimise_df, round(1 / (len(optimise_df.columns) * 2), 3)
                    ),
                    index=pd.Series(0),
                )
            except Exception:
                print(f"Max Sharpe Opt Failed For {start}")
                weights = pd.DataFrame(
                    [
                        1 / len(optimise_df.columns)
                        for i in range(len(optimise_df.columns))
                    ],
                    index=optimise_df.columns.tolist(),
                    columns=pd.Series(0),
                ).T
            temp_df: pd.DataFrame = returns_df[start:end_date]

            temp_df = (
                temp_df.stack()
                .to_frame("return")
                .reset_index(level=0)
                .merge(
                    weights.stack().to_frame("weight").reset_index(level=0, drop=True),
                    left_index=True,
                    right_index=True,
                )
                .reset_index()
                .set_index(["Date", "Ticker"])
                .unstack()
                .stack()
            )
            temp_df.index.names = ["date", "ticker"]
            temp_df["weighted_return"] = temp_df["return"] * temp_df["weight"]
            temp_df = (
                temp_df.groupby(level=0)["weighted_return"]
                .sum()
                .to_frame("Strategy Return")
            )
            portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
        except Exception as e:
            pass

    return portfolio_df


@dispatch()
def portfolio_optimization() -> pd.DataFrame:
    return pd.read_csv("optimised_data.txt", parse_dates=["date"]).set_index("date")


@dispatch(dict, pd.DataFrame)
def portfolio_optimization(fixed_dates, cluster_data) -> pd.DataFrame:
    """"""

    prices = get_daily_prices(cluster_data)  # now new_df
    portfolio_returns = monthly_optimise(prices, fixed_dates)
    return portfolio_returns.drop_duplicates()


def get_spy500_data():
    """"""
    spy: pd.DataFrame = yf.download(
        tickers="SPY", start="2015-01-01", end=datetime.date.today()
    )
    spy_ret = (
        np.log(spy[["Adj Close"]])
        .diff()
        .dropna()
        .rename({"Adj Close": "SPY Buy&Hold"}, axis=1)
    )
    return spy_ret


def visualise_and_compare_sp500(opt_df):
    """"""
    spy_ret = get_spy500_data()
    port_df = opt_df.merge(spy_ret, left_index=True, right_index=True)
    port_cum_returns = np.exp(np.log1p(port_df).cumsum()) - 1
    port_cum_returns.plot(figsize=(16, 6))
    plt.title("Unsupervised Learning Trading Strategy Returns Over Time")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.ylabel("Return")
    plt.show()
    return port_df


if __name__ == "__main__":
    feature_data = get_feature_data()
    cluster_data = fit_k_means(feature_data)
    fixed_dates = get_dates_and_tickers(cluster_data)
    optimised_df = portfolio_optimization(fixed_dates, cluster_data)
    # optimised_df = portfolio_optimization()
    _x = visualise_and_compare_sp500(optimised_df)
