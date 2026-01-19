import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class FreqAIStrategy(IStrategy):
    """
    Swing Trading Strategy for BTC/USDT on 1-hour timeframe.

    Targets larger moves (5-15%) with fewer, higher-quality trades.
    Holds positions for days to weeks rather than minutes to hours.
    """
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # Swing trading ROI - let winners run
    minimal_roi = {
        "0": 0.15,      # 15% take profit initially
        "72": 0.10,     # 10% after 3 days
        "168": 0.07,    # 7% after 1 week
        "336": 0.05     # 5% after 2 weeks
    }

    # Wider stoploss for swing trading
    stoploss = -0.07  # 7% stoploss

    # Trailing stop to lock in profits
    trailing_stop = True
    trailing_stop_positive = 0.03      # Start trailing at 3% profit
    trailing_stop_positive_offset = 0.05  # Only activate after 5% profit
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False

    # Need more candles for longer-term indicators
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend EMAs (longer periods for swing trading)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # RSI for momentum
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # MACD for trend confirmation
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Bollinger Bands for volatility and mean reversion
        bb = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe["bb_upper"] = bb["upperband"]
        dataframe["bb_middle"] = bb["middleband"]
        dataframe["bb_lower"] = bb["lowerband"]

        # ADX for trend strength
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # ATR for volatility (useful for position sizing)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Weekly trend (using 168-hour = 1 week lookback)
        dataframe["ema_weekly"] = ta.EMA(dataframe, timeperiod=168)

        # Volume analysis
        dataframe["volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Trend following: price above major EMAs with momentum
                (dataframe["close"] > dataframe["ema_50"]) &
                (dataframe["ema_20"] > dataframe["ema_50"]) &
                (dataframe["rsi"] > 40) &
                (dataframe["rsi"] < 70) &
                (dataframe["macd"] > dataframe["macdsignal"]) &
                (dataframe["adx"] > 20) &  # Trend strength filter
                (dataframe["volume"] > dataframe["volume_sma"] * 0.8)
            ) |
            (
                # Pullback entry: strong trend with temporary dip
                (dataframe["close"] > dataframe["ema_100"]) &
                (dataframe["close"] < dataframe["ema_20"]) &  # Pulled back to EMA
                (dataframe["rsi"] < 45) &
                (dataframe["rsi"] > 30) &
                (dataframe["adx"] > 25) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                # Overbought exit
                (dataframe["rsi"] > 75) &
                (dataframe["close"] > dataframe["bb_upper"])
            ) |
            (
                # Trend reversal: price breaks below key support
                (dataframe["ema_20"] < dataframe["ema_50"]) &
                (dataframe["close"] < dataframe["ema_50"]) &
                (dataframe["macd"] < dataframe["macdsignal"])
            ) |
            (
                # Momentum loss
                (dataframe["rsi"] < 40) &
                (dataframe["macd"] < 0) &
                (dataframe["close"] < dataframe["ema_20"])
            ),
            "exit_long"
        ] = 1
        return dataframe
