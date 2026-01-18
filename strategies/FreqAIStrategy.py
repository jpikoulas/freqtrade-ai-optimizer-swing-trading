import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class FreqAIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False

    # ROI settings
    minimal_roi = {
        "0": 0.02,
        "30": 0.01,
        "60": 0.005
    }

    stoploss = -0.03

    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    startup_candle_count: int = 50

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs for trend
        dataframe["ema_8"] = ta.EMA(dataframe, timeperiod=8)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Bollinger Bands
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_lower"] = bb["lowerband"]

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["stoch_k"] = stoch["slowk"]
        dataframe["stoch_d"] = stoch["slowd"]

        # Momentum indicators
        dataframe["rsi_momentum"] = dataframe["rsi"] > dataframe["rsi"].shift(1)
        dataframe["macd_momentum"] = dataframe["macdhist"] > dataframe["macdhist"].shift(1)

        # Volume
        dataframe["volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Trend-following setup
                (dataframe["ema_8"] > dataframe["ema_21"]) &
                (dataframe["close"] > dataframe["ema_8"]) &
                (dataframe["rsi"] > 40) &
                (dataframe["rsi"] < 65) &
                (dataframe["macd"] > dataframe["macdsignal"]) &
                (dataframe["volume"] > 0)
            ) |
            (
                # Oversold bounce
                (dataframe["rsi"] < 35) &
                (dataframe["rsi_momentum"] == True) &
                (dataframe["close"] < dataframe["bb_lower"]) &
                (dataframe["stoch_k"] < 25) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Overbought exit
                (dataframe["rsi"] > 70) &
                (dataframe["close"] > dataframe["bb_upper"])
            ) |
            (
                # Trend reversal
                (dataframe["ema_8"] < dataframe["ema_21"]) &
                (dataframe["macd"] < dataframe["macdsignal"]) &
                (dataframe["rsi"] < 50)
            ),
            "exit_long"
        ] = 1
        return dataframe
