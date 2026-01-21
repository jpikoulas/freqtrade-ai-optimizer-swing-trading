#!/usr/bin/env python3
"""
FreqTrade AI Strategy Optimizer - Local Runner with Hyperopt
Uses hyperopt for parameter optimization, then DeepSeek for strategy improvements.

Features:
- Crash recovery: Saves state after each iteration
- Graceful shutdown: Saves best strategy on SIGTERM/SIGINT
- Resume from best: Automatically resumes from best strategy on restart
"""
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add optimizer to path
sys.path.insert(0, str(Path(__file__).parent / "optimizer"))

from openai import OpenAI

# Configuration - support both local and Docker execution
PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).parent))
USER_DATA_DIR = PROJECT_DIR / "user_data"
CONFIG_PATH = PROJECT_DIR / "config" / "freqtrade_config.json"
STRATEGIES_DIR = PROJECT_DIR / "strategies"
RESULTS_DIR = USER_DATA_DIR / "backtest_results"
HYPEROPT_DIR = USER_DATA_DIR / "hyperopt_results"

# Host paths for Docker volume mounts (when running optimizer in Docker)
HOST_PROJECT_DIR = Path(os.environ.get("HOST_PROJECT_DIR", PROJECT_DIR))

MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30"))
TARGET_PROFIT = float(os.environ.get("TARGET_PROFIT", "5.0"))
BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", "90"))
HYPEROPT_EPOCHS = int(os.environ.get("HYPEROPT_EPOCHS", "100"))

# DeepSeek model to use - deepseek-chat is more reliable for code generation
# deepseek-reasoner can overthink and generate overly complex conditions
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# Slack webhook URL for notifications (optional)
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

# Consecutive zero-trade threshold before resetting to base strategy
ZERO_TRADE_RESET_THRESHOLD = 2

# State file for crash recovery
STATE_FILE = USER_DATA_DIR / ".optimizer_state.json"

# Global state for signal handlers
_shutdown_requested = False
_current_best_strategy = None
_current_best_profit = float('-inf')


def send_slack_notification(message: str, emoji: str = ":chart_with_upwards_trend:"):
    """Send a notification to Slack via webhook."""
    if not SLACK_WEBHOOK_URL:
        return  # Slack not configured, silently skip

    import urllib.request
    import urllib.error

    payload = {
        "text": message,
        "icon_emoji": emoji,
        "username": "FreqTrade Swing Optimizer"
    }

    try:
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                print("  Slack notification sent")
            else:
                print(f"  Slack notification failed: {response.status}")
    except urllib.error.URLError as e:
        print(f"  Slack notification failed: {e}")
    except Exception as e:
        print(f"  Slack notification error: {e}")


def notify_new_best(results: dict, iteration: int, previous_best: float):
    """Send Slack notification for new best result."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    emoji = ":rocket:" if results["total_profit_pct"] >= 5 else ":chart_with_upwards_trend:" if results["total_profit_pct"] >= 0 else ":chart_with_downwards_trend:"

    message = f"""*:dart: New Best SWING Strategy Found!*
*Time:* {timestamp}
*Iteration:* {iteration}

```
Results:
  Total Trades: {results['total_trades']}
  Total Profit: {results['total_profit_pct']:.2f}%
  Win Rate: {results['win_rate']:.2f}%
  Max Drawdown: {results['max_drawdown_pct']:.2f}%
  Sharpe Ratio: {results['sharpe_ratio']:.2f}
  Avg Profit/Trade: {results['avg_profit_pct']:.2f}%
  *** New best: {results['total_profit_pct']:.2f}% ***
```
_Previous best: {previous_best:.2f}%_"""

    send_slack_notification(message, emoji)


def notify_target_achieved(results: dict, iteration: int):
    """Send Slack notification when target profit is achieved."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = f"""*:trophy: SWING TARGET PROFIT ACHIEVED! :tada:*
*Time:* {timestamp}
*Iteration:* {iteration}

```
Final Results:
  Total Trades: {results['total_trades']}
  Total Profit: {results['total_profit_pct']:.2f}%
  Win Rate: {results['win_rate']:.2f}%
  Max Drawdown: {results['max_drawdown_pct']:.2f}%
  Sharpe Ratio: {results['sharpe_ratio']:.2f}
  Avg Profit/Trade: {results['avg_profit_pct']:.2f}%
```
*Target was: {TARGET_PROFIT}%*
:star: Optimization complete!"""

    send_slack_notification(message, ":trophy:")


def notify_optimizer_started():
    """Send Slack notification when optimizer starts."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = f"""*:rocket: FreqTrade SWING Optimizer Started*
*Time:* {timestamp}

```
Configuration:
  Target Profit: {TARGET_PROFIT}%
  Max Iterations: {MAX_ITERATIONS}
  Backtest Days: {BACKTEST_DAYS}
  Hyperopt Epochs: {HYPEROPT_EPOCHS}
  AI Model: {DEEPSEEK_MODEL}
  Timeframe: 1h (Swing Trading)
```"""

    send_slack_notification(message, ":rocket:")


def load_optimizer_state():
    """Load optimizer state from crash recovery file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            print(f"Loaded state from {STATE_FILE}")
            return state
        except Exception as e:
            print(f"Could not load state: {e}")
    return None


def save_optimizer_state(iteration, best_profit, best_strategy, iteration_results, consecutive_zero_trades=0):
    """Save optimizer state for crash recovery."""
    global _current_best_strategy, _current_best_profit
    _current_best_strategy = best_strategy
    _current_best_profit = best_profit

    state = {
        "iteration": iteration,
        "best_profit": best_profit,
        "consecutive_zero_trades": consecutive_zero_trades,
        "iteration_results": iteration_results,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save state: {e}")


def save_best_strategy_immediately(strategy, profit, reason="shutdown"):
    """Save the best strategy immediately (used during shutdown)."""
    if strategy is None:
        return

    backup_dir = USER_DATA_DIR / "strategy_backups"
    backup_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"FreqAIStrategy_{reason}_{profit:.1f}pct_{timestamp}.py"
    filepath = backup_dir / filename

    try:
        with open(filepath, 'w') as f:
            f.write(strategy)
        print(f"Saved best strategy to {filepath}")
    except Exception as e:
        print(f"Warning: Could not save strategy: {e}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    signal_name = signal.Signals(signum).name
    print(f"\n{'='*60}")
    print(f"Received {signal_name} - initiating graceful shutdown...")
    print(f"{'='*60}")

    _shutdown_requested = True

    # Save best strategy immediately
    if _current_best_strategy:
        save_best_strategy_immediately(_current_best_strategy, _current_best_profit, "graceful_stop")

    print("Shutdown complete. Best strategy saved.")
    sys.exit(0)


def cleanup_handler():
    """Cleanup handler called on exit."""
    if _current_best_strategy and not _shutdown_requested:
        print("Saving best strategy on exit...")
        save_best_strategy_immediately(_current_best_strategy, _current_best_profit, "exit")


def check_and_download_data():
    """Check if data exists and download if needed."""
    data_dir = USER_DATA_DIR / "data" / "binance"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing data files
    data_files = list(data_dir.glob("*.json.gz")) + list(data_dir.glob("*.json"))
    if data_files:
        print(f"Found {len(data_files)} existing data files")
        return True

    print("No data found, downloading...")
    return download_data()


def download_data():
    """Download historical data using freqtrade."""
    timerange = get_timerange()

    cmd = [
        "docker", "compose", "run", "--rm",
        "freqtrade",
        "download-data",
        "--config", "/freqtrade/config/freqtrade_config.json",
        "--timerange", timerange,
        "--timeframe", "1h",  # Swing trading uses 1h
    ]

    print(f"Downloading data for timerange {timerange}...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                                cwd=PROJECT_DIR, env=get_docker_env())
        if result.returncode != 0:
            print(f"Data download failed: {result.stderr}")
            return False
        print("Data download complete")
        return True
    except subprocess.TimeoutExpired:
        print("Data download timed out")
        return False
    except Exception as e:
        print(f"Error downloading data: {e}")
        return False


def get_timerange():
    """Calculate timerange for backtesting."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_DAYS)
    return f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"


def get_docker_env():
    """Get environment variables for docker compose subprocess calls."""
    env = os.environ.copy()
    # Ensure HOST_PROJECT_DIR is set for docker-compose.yml volume paths
    env["HOST_PROJECT_DIR"] = str(HOST_PROJECT_DIR)
    return env


def run_hyperopt(strategy_name="FreqAIStrategy", epochs=100):
    """Run hyperopt optimization using Docker Compose."""
    timerange = get_timerange()

    # Container runs as root, no sudo needed. Pass HOST_PROJECT_DIR for volume paths
    cmd = [
        "docker", "compose", "run", "--rm",
        "freqtrade",
        "hyperopt",
        "--config", "/freqtrade/config/freqtrade_config.json",
        "--strategy", strategy_name,
        "--hyperopt-loss", "SharpeHyperOptLoss",
        "--timerange", timerange,
        "--epochs", str(epochs),
        "--spaces", "roi", "stoploss", "trailing",
        "--min-trades", "10",  # Lower for swing trading (fewer trades expected)
        "-j", "4"  # Use 4 parallel jobs
    ]

    print(f"Running hyperopt with {epochs} epochs...")

    try:
        # Use PROJECT_DIR as cwd and pass HOST_PROJECT_DIR env for volume paths
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                                cwd=PROJECT_DIR, env=get_docker_env())
        if result.returncode != 0:
            print(f"Hyperopt failed: {result.stderr}")
            return None, result.stderr
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return True, result.stdout
    except subprocess.TimeoutExpired:
        return None, "Hyperopt timed out"
    except Exception as e:
        return None, str(e)


def run_backtest(strategy_name="FreqAIStrategy"):
    """Run backtest using Docker Compose."""
    timerange = get_timerange()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    cmd = [
        "docker", "compose", "run", "--rm",
        "freqtrade",
        "backtesting",
        "--config", "/freqtrade/config/freqtrade_config.json",
        "--strategy", strategy_name,
        "--timerange", timerange,
        "--cache", "none",  # Disable cache to ensure fresh results
        "--export", "trades",
        "--export-filename", f"/freqtrade/user_data/backtest_results/backtest-result-{timestamp}.json"
    ]

    print(f"Running backtest...")

    try:
        # Use PROJECT_DIR as cwd and pass HOST_PROJECT_DIR env for volume paths
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                                cwd=PROJECT_DIR, env=get_docker_env())
        if result.returncode != 0:
            print(f"Backtest failed: {result.stderr}")
            return None, result.stderr
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return True, None
    except subprocess.TimeoutExpired:
        return None, "Backtest timed out"
    except Exception as e:
        return None, str(e)


def apply_hyperopt_results():
    """Apply best hyperopt results to strategy."""
    hyperopt_files = list(HYPEROPT_DIR.glob("*.pickle")) + list(HYPEROPT_DIR.glob("*.fthypt"))
    if not hyperopt_files:
        print("No hyperopt results found")
        return None

    cmd = [
        "docker", "compose", "run", "--rm",
        "freqtrade",
        "hyperopt-show",
        "--config", "/freqtrade/config/freqtrade_config.json",
        "--best",
        "--no-header",
        "--print-json"
    ]

    try:
        # Use PROJECT_DIR as cwd and pass HOST_PROJECT_DIR env for volume paths
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                cwd=PROJECT_DIR, env=get_docker_env())
        if result.returncode == 0 and result.stdout.strip():
            try:
                best_params = json.loads(result.stdout.strip())
                return best_params
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^{}]*\}', result.stdout)
                if json_match:
                    return json.loads(json_match.group())
        print(f"Could not get hyperopt results: {result.stderr}")
        return None
    except Exception as e:
        print(f"Error getting hyperopt results: {e}")
        return None


def find_latest_results():
    """Find the most recent backtest results file."""
    last_result_file = RESULTS_DIR / ".last_result.json"
    if last_result_file.exists():
        try:
            with open(last_result_file) as f:
                data = json.load(f)
            latest = data.get("latest_backtest")
            if latest:
                return RESULTS_DIR / latest
        except:
            pass

    results_files = list(RESULTS_DIR.glob("backtest-result-*.zip"))
    if not results_files:
        return None
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return results_files[0]


def parse_results(results_path):
    """Parse backtest results from zip file."""
    import zipfile

    try:
        with zipfile.ZipFile(results_path) as zf:
            json_files = [f for f in zf.namelist() if f.endswith('.json') and not f.endswith('_config.json')]
            if not json_files:
                print(f"No JSON file found in {results_path}")
                return None

            with zf.open(json_files[0]) as f:
                data = json.load(f)

        strategy_results = None
        if "strategy" in data:
            for name, results in data["strategy"].items():
                strategy_results = results
                break

        if strategy_results is None:
            print(f"Could not find strategy results in {results_path}")
            return None

        return {
            "total_profit_pct": strategy_results.get("profit_total", 0) * 100,
            "win_rate": strategy_results.get("winrate", 0) * 100,
            "total_trades": strategy_results.get("total_trades", 0),
            "max_drawdown_pct": abs(strategy_results.get("max_drawdown_account", 0) * 100),
            "avg_profit_pct": strategy_results.get("profit_mean", 0) * 100 if strategy_results.get("profit_mean") else 0,
            "sharpe_ratio": strategy_results.get("sharpe", 0),
            "profit_factor": strategy_results.get("profit_factor", 0),
        }
    except Exception as e:
        print(f"Error parsing results: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_base_strategy():
    """Return a proven base swing trading strategy that generates trades.

    This strategy uses trend-following conditions on 1h timeframe.
    Targets fewer, higher-quality trades with larger profit targets.
    """
    return '''import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class FreqAIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = False

    # Swing trading ROI - let winners run
    minimal_roi = {"0": 0.12, "72": 0.08, "168": 0.05, "336": 0.03}
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 100

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_100"] = ta.EMA(dataframe, timeperiod=100)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Trend following entry with confirmation
        dataframe.loc[
            (
                (dataframe["close"] > dataframe["ema_50"]) &
                (dataframe["ema_20"] > dataframe["ema_50"]) &
                (dataframe["rsi"] > 40) &
                (dataframe["rsi"] < 70) &
                (dataframe["macd"] > dataframe["macdsignal"]) &
                (dataframe["volume"] > 0)
            ) |
            (
                # Pullback entry in uptrend
                (dataframe["close"] > dataframe["ema_100"]) &
                (dataframe["close"] < dataframe["ema_20"]) &
                (dataframe["rsi"] < 45) &
                (dataframe["rsi"] > 30) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit on overbought or trend reversal
        dataframe.loc[
            (dataframe["rsi"] > 75) |
            (
                (dataframe["ema_20"] < dataframe["ema_50"]) &
                (dataframe["close"] < dataframe["ema_50"]) &
                (dataframe["macd"] < dataframe["macdsignal"])
            ),
            "exit_long"
        ] = 1
        return dataframe
'''


def generate_improved_strategy(client, current_strategy, results, iteration, hyperopt_params=None, consecutive_zero_trades=0):
    """Generate improved strategy using DeepSeek, incorporating hyperopt results."""

    # Build analysis based on results
    issues = []
    suggestions = []

    # Determine priority based on current state - SWING TRADING expectations
    # On 1h timeframe over 90 days, expect 10-50 trades (not 200+)
    if results["total_trades"] == 0:
        issues.append("CRITICAL: Zero trades generated!")
        suggestions.append("Your entry conditions are TOO STRICT. You MUST simplify them.")
        suggestions.append("Use 3-4 simple conditions: EMA trend + RSI range + MACD signal + volume > 0")
        suggestions.append("DO NOT use .shift() comparisons - they are too restrictive")
        suggestions.append("Widen RSI range: use (rsi > 35) & (rsi < 70) instead of narrow ranges")
    elif results["total_trades"] < 5:
        issues.append(f"Too few trades ({results['total_trades']}) - need at least 10 for swing trading statistics")
        suggestions.append("Loosen entry conditions - widen RSI thresholds or add alternative entry (pullback)")
    elif results["total_trades"] > 100:
        issues.append(f"Too many trades ({results['total_trades']}) - overtrading for swing strategy")
        suggestions.append("Add stronger trend filter (ADX > 25) or require price above ema_100")
    elif results["total_profit_pct"] < 0:
        issues.append(f"Strategy losing money ({results['total_profit_pct']:.2f}%)")
        if results["win_rate"] < 45:
            suggestions.append(f"Low win rate ({results['win_rate']:.1f}%) - add trend confirmation (ema_20 > ema_50)")
        else:
            suggestions.append("Exits may be too early - let winners run longer, raise RSI exit to 75+")
    else:
        issues.append(f"Strategy profitable at {results['total_profit_pct']:.2f}% - fine-tune for better performance")
        if results["win_rate"] < 50:
            suggestions.append("Add ADX filter (adx > 20) to improve trade quality")

    if results["max_drawdown_pct"] > 15:
        issues.append(f"High drawdown ({results['max_drawdown_pct']:.2f}%)")
        suggestions.append("Consider adding trend filter or tighter trailing stop")

    analysis = "\n".join(f"- {i}" for i in issues)
    suggestion_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "- Continue current approach"

    # Include hyperopt params if available
    hyperopt_context = ""
    if hyperopt_params:
        hyperopt_context = f"""
HYPEROPT OPTIMIZED PARAMETERS (use these exact values):
{json.dumps(hyperopt_params, indent=2)}

Apply these to minimal_roi, stoploss, and trailing_stop settings.
"""

    # Build system prompt - SWING TRADING on 1h timeframe
    system_prompt = """You are an expert FreqTrade SWING TRADING strategy developer. Your task is to generate a WORKING 1-hour timeframe strategy.

SWING TRADING PRINCIPLES:
- Timeframe: 1h (MUST use timeframe = "1h")
- Target: 5-15% profit per trade, held for days to weeks
- Fewer trades (10-50 over 90 days) but higher quality
- Let winners run with trailing stops
- Use longer EMAs (20, 50, 100, 200) for trend identification

ABSOLUTE PRIORITY #1: THE STRATEGY MUST GENERATE TRADES!
A strategy with 0 trades is USELESS. Better to have losing trades than no trades.

CODE REQUIREMENTS:
1. Return ONLY valid Python code - no explanations, no markdown, no ```
2. Class name MUST be 'FreqAIStrategy'
3. timeframe MUST be "1h"
4. Every indicator used in entry/exit MUST be defined in populate_indicators()
5. Use ONLY: ta.RSI, ta.MACD, ta.EMA, ta.SMA, ta.ATR, ta.ADX, ta.BBANDS
6. BBANDS must use floats: ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

ENTRY CONDITIONS - SWING TRADING:
- Use 3-5 conditions connected with &
- Use | (OR) for alternative entries (trend following + pullback)
- GOOD: (close > ema_50) & (ema_20 > ema_50) & (rsi > 40) & (rsi < 70) & (macd > macdsignal)
- BAD: More than 6 conditions = zero trades
- BAD: Using .shift() comparisons = often zero trades
- BAD: Narrow RSI ranges like (rsi > 50) & (rsi < 55) = zero trades

WORKING SWING ENTRY EXAMPLE:
    dataframe.loc[
        (
            # Trend following
            (dataframe["close"] > dataframe["ema_50"]) &
            (dataframe["ema_20"] > dataframe["ema_50"]) &
            (dataframe["rsi"] > 40) & (dataframe["rsi"] < 70) &
            (dataframe["macd"] > dataframe["macdsignal"]) &
            (dataframe["volume"] > 0)
        ) |
        (
            # Pullback entry
            (dataframe["close"] > dataframe["ema_100"]) &
            (dataframe["close"] < dataframe["ema_20"]) &
            (dataframe["rsi"] < 45) & (dataframe["rsi"] > 30) &
            (dataframe["volume"] > 0)
        ),
        "enter_long"
    ] = 1

WORKING SWING EXIT EXAMPLE:
    dataframe.loc[
        (dataframe["rsi"] > 75) |
        (
            (dataframe["ema_20"] < dataframe["ema_50"]) &
            (dataframe["close"] < dataframe["ema_50"]) &
            (dataframe["macd"] < dataframe["macdsignal"])
        ),
        "exit_long"
    ] = 1

DEFAULT SWING SETTINGS:
    minimal_roi = {"0": 0.12, "72": 0.08, "168": 0.05, "336": 0.03}
    stoploss = -0.06
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    startup_candle_count = 100"""

    # Build user prompt with clear instructions
    zero_trade_warning = ""
    if results["total_trades"] == 0:
        zero_trade_warning = """
*** URGENT: YOUR LAST STRATEGY GENERATED ZERO TRADES! ***
This means the entry conditions were impossible to satisfy.
You MUST simplify the entry conditions dramatically.
Remove any .shift() comparisons and widen all thresholds.
"""

    user_prompt = f"""Current strategy:

```python
{current_strategy}
```

BACKTEST RESULTS:
- Total Trades: {results['total_trades']}
- Total Profit: {results['total_profit_pct']:.2f}%
- Win Rate: {results['win_rate']:.2f}%
- Max Drawdown: {results['max_drawdown_pct']:.2f}%
- Sharpe Ratio: {results['sharpe_ratio']:.2f}
{zero_trade_warning}
ISSUES:
{analysis}

WHAT TO DO:
{suggestion_text}
{hyperopt_context}
TARGET: Generate 10-50 swing trades with positive profit over 90 days on 1h timeframe.

Return ONLY the complete Python code. No explanations."""

    print(f"Generating improved strategy (iteration {iteration}) using DeepSeek...")

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=8000,
            temperature=0.5
        )

        response_text = response.choices[0].message.content

        # Clean up response
        import re
        response_text = re.sub(r'^```python\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()

        # Fix common TA-Lib issues
        response_text = re.sub(r'nbdevup=(\d+)(?!\.)', r'nbdevup=\1.0', response_text)
        response_text = re.sub(r'nbdevdn=(\d+)(?!\.)', r'nbdevdn=\1.0', response_text)

        # Validate syntax
        try:
            compile(response_text, '<string>', 'exec')
            print("Generated strategy code validated successfully")
        except SyntaxError as e:
            print(f"Generated code has syntax error: {e}")
            return None

        return response_text
    except Exception as e:
        print(f"Error generating strategy: {e}")
        return None


def main():
    global _current_best_strategy, _current_best_profit

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(cleanup_handler)

    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set")
        print("Please set DEEPSEEK_API_KEY in your .env file")
        sys.exit(1)

    # Initialize DeepSeek client (OpenAI-compatible API)
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    # Ensure directories exist
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HYPEROPT_DIR.mkdir(parents=True, exist_ok=True)
    (USER_DATA_DIR / "strategies").mkdir(parents=True, exist_ok=True)
    (USER_DATA_DIR / "strategy_backups").mkdir(parents=True, exist_ok=True)

    # Check and download data if needed
    if not check_and_download_data():
        print("WARNING: Could not verify data. Backtest may fail.")

    # Check for existing state (crash recovery)
    saved_state = load_optimizer_state()
    start_iteration = 1
    best_profit = float('-inf')
    best_strategy = None
    iteration_results = []
    consecutive_zero_trades = 0

    if saved_state:
        print(f"\n{'='*60}")
        print("RESUMING FROM SAVED STATE")
        print(f"{'='*60}")
        start_iteration = saved_state.get("iteration", 0) + 1
        best_profit = saved_state.get("best_profit", float('-inf'))
        iteration_results = saved_state.get("iteration_results", [])
        consecutive_zero_trades = saved_state.get("consecutive_zero_trades", 0)
        print(f"Resuming from iteration {start_iteration}")
        print(f"Previous best profit: {best_profit:.2f}%")
        print(f"{'='*60}\n")

    # Load best strategy if exists, otherwise copy initial
    best_backup = USER_DATA_DIR / "strategy_backups"
    best_files = sorted(best_backup.glob("FreqAIStrategy_best_*.py"), key=lambda x: x.stat().st_mtime, reverse=True)
    if best_files and saved_state:
        print(f"Loading best strategy from {best_files[0]}")
        with open(best_files[0]) as f:
            best_strategy = f.read()
        # Use best strategy as current
        with open(USER_DATA_DIR / "strategies" / "FreqAIStrategy.py", 'w') as f:
            f.write(best_strategy)
    else:
        # Copy initial strategy
        initial_strategy = STRATEGIES_DIR / "FreqAIStrategy.py"
        shutil.copy2(initial_strategy, USER_DATA_DIR / "strategies" / "FreqAIStrategy.py")

    print("=" * 60)
    print("FreqTrade SWING TRADING Optimizer (Hyperopt + DeepSeek)")
    print("=" * 60)
    print(f"Timeframe: 1h (Swing Trading)")
    print(f"Target profit: {TARGET_PROFIT}%")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Hyperopt epochs per iteration: {HYPEROPT_EPOCHS}")
    print(f"AI Model: DeepSeek ({DEEPSEEK_MODEL})")
    if saved_state:
        print(f"Resuming from: iteration {start_iteration}")
    print("=" * 60)

    # Send startup notification
    notify_optimizer_started()

    # Load current strategy
    with open(USER_DATA_DIR / "strategies" / "FreqAIStrategy.py") as f:
        current_strategy = f.read()

    if best_strategy is None:
        best_strategy = current_strategy

    # Update global state for signal handlers
    _current_best_strategy = best_strategy
    _current_best_profit = best_profit

    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
        # Check for shutdown request
        if _shutdown_requested:
            print("Shutdown requested, exiting loop...")
            break

        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{MAX_ITERATIONS}")
        print(f"{'='*60}")

        # Step 1: Run backtest FIRST (skip hyperopt if 0 trades - it can't help)
        print("\nStep 1: Running backtest...")
        success, error = run_backtest()

        if not success:
            print(f"Backtest failed: {error}")
            continue

        # Parse results
        results_path = find_latest_results()
        if not results_path:
            print("No results found")
            continue

        results = parse_results(results_path)
        if not results:
            print("Failed to parse results")
            continue

        # Log results
        print(f"\nResults:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Total Profit: {results['total_profit_pct']:.2f}%")
        print(f"  Win Rate: {results['win_rate']:.2f}%")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")

        # Track consecutive zero trades
        if results["total_trades"] == 0:
            consecutive_zero_trades += 1
            print(f"  WARNING: Zero trades! (consecutive: {consecutive_zero_trades})")
        else:
            consecutive_zero_trades = 0  # Reset counter

        # Check if we need to reset to base strategy
        if consecutive_zero_trades >= ZERO_TRADE_RESET_THRESHOLD:
            print(f"\n*** RESETTING to base strategy after {consecutive_zero_trades} consecutive zero-trade iterations ***")
            current_strategy = get_base_strategy()
            consecutive_zero_trades = 0
            # Save and continue
            with open(USER_DATA_DIR / "strategies" / "FreqAIStrategy.py", 'w') as f:
                f.write(current_strategy)
            continue

        iteration_results.append({
            "iteration": iteration,
            **results,
            "hyperopt_params": None,
            "timestamp": datetime.now().isoformat()
        })

        # Track best (only if we have trades)
        if results["total_trades"] > 0 and results["total_profit_pct"] > best_profit:
            previous_best = best_profit
            best_profit = results["total_profit_pct"]
            best_strategy = current_strategy
            _current_best_strategy = best_strategy
            _current_best_profit = best_profit
            print(f"  *** New best: {best_profit:.2f}% ***")

            # Send Slack notification for new best
            notify_new_best(results, iteration, previous_best)

            # Save best strategy immediately
            backup_dir = USER_DATA_DIR / "strategy_backups"
            with open(backup_dir / f"FreqAIStrategy_best_{best_profit:.1f}pct.py", 'w') as f:
                f.write(best_strategy)

        # Check target - lower trade requirement for swing trading
        if results["total_profit_pct"] >= TARGET_PROFIT and results["total_trades"] >= 10:
            print(f"\n{'='*60}")
            print(f"TARGET ACHIEVED! Profit: {results['total_profit_pct']:.2f}%")
            print(f"{'='*60}")

            # Send Slack notification for target achieved
            notify_target_achieved(results, iteration)

            # Save winning strategy to backup directory
            backup_dir = USER_DATA_DIR / "strategy_backups"
            backup_dir.mkdir(exist_ok=True)
            with open(backup_dir / f"FreqAIStrategy_winning_{results['total_profit_pct']:.1f}pct.py", 'w') as f:
                f.write(current_strategy)
            break

        # Step 2: Run hyperopt ONLY if we have trades (otherwise it's useless)
        hyperopt_params = None
        if results["total_trades"] >= 10:  # Lower threshold for swing trading
            print("\nStep 2: Running hyperopt for parameter optimization...")
            success, output = run_hyperopt(epochs=HYPEROPT_EPOCHS)
            if success:
                hyperopt_params = apply_hyperopt_results()
                if hyperopt_params:
                    print(f"Hyperopt found optimal parameters: {hyperopt_params}")
        else:
            print("\nStep 2: Skipping hyperopt (need trades first)")

        # Step 3: Generate improved strategy using DeepSeek
        print("\nStep 3: Generating improved strategy with DeepSeek...")
        improved_strategy = generate_improved_strategy(
            client, current_strategy, results, iteration, hyperopt_params, consecutive_zero_trades
        )

        if improved_strategy:
            current_strategy = improved_strategy
            # Save iteration backup to separate directory (not strategies/ to avoid class name conflicts)
            backup_dir = USER_DATA_DIR / "strategy_backups"
            backup_dir.mkdir(exist_ok=True)
            with open(backup_dir / f"FreqAIStrategy_iter{iteration:02d}.py", 'w') as f:
                f.write(current_strategy)
            # Update main strategy (the only file in strategies/ with this class)
            with open(USER_DATA_DIR / "strategies" / "FreqAIStrategy.py", 'w') as f:
                f.write(current_strategy)

        # Save state for crash recovery after each iteration
        save_optimizer_state(iteration, best_profit, best_strategy, iteration_results, consecutive_zero_trades)

    # Save report
    report = {
        "summary": {
            "target_profit": TARGET_PROFIT,
            "max_iterations": MAX_ITERATIONS,
            "hyperopt_epochs": HYPEROPT_EPOCHS,
            "ai_model": f"deepseek/{DEEPSEEK_MODEL}",
            "iterations_run": len(iteration_results),
            "best_profit": best_profit,
        },
        "iterations": iteration_results,
    }

    with open(USER_DATA_DIR / "optimization_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Save best strategy to backup directory
    backup_dir = USER_DATA_DIR / "strategy_backups"
    backup_dir.mkdir(exist_ok=True)
    with open(backup_dir / f"FreqAIStrategy_best_{best_profit:.1f}pct.py", 'w') as f:
        f.write(best_strategy)

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"Best profit achieved: {best_profit:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
