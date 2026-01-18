#!/usr/bin/env python3
"""
FreqTrade AI Strategy Optimizer - Local Runner with Hyperopt
Uses hyperopt for parameter optimization, then DeepSeek for strategy improvements.
"""
import json
import os
import shutil
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

# Consecutive zero-trade threshold before resetting to base strategy
ZERO_TRADE_RESET_THRESHOLD = 2


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
        "--min-trades", "20",
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
    """Return a proven base strategy that generates trades.

    This strategy uses simple RSI-based conditions that reliably trigger
    trades in most market conditions. The AI will improve upon this foundation.
    """
    return '''import talib.abstract as ta
from pandas import DataFrame
from freqtrade.strategy import IStrategy


class FreqAIStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = False

    # Loose ROI targets - let trades run
    minimal_roi = {"0": 0.03, "60": 0.02, "120": 0.01}
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count: int = 30

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Simple entry: RSI recovering from oversold OR price above trend
        dataframe.loc[
            (
                (dataframe["rsi"] < 45) &
                (dataframe["rsi"] > 25) &
                (dataframe["volume"] > 0)
            ) |
            (
                (dataframe["close"] > dataframe["ema_20"]) &
                (dataframe["rsi"] < 60) &
                (dataframe["volume"] > 0)
            ),
            "enter_long"
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit on overbought OR below trend
        dataframe.loc[
            (dataframe["rsi"] > 70) |
            (
                (dataframe["close"] < dataframe["ema_50"]) &
                (dataframe["rsi"] < 40)
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

    # Determine priority based on current state
    if results["total_trades"] == 0:
        issues.append("CRITICAL: Zero trades generated!")
        suggestions.append("Your entry conditions are TOO STRICT. You MUST simplify them.")
        suggestions.append("Use ONLY 3-4 simple conditions: EMA crossover + RSI range + MACD signal + volume > 0")
        suggestions.append("DO NOT use .shift() comparisons or momentum indicators - they are too restrictive")
        suggestions.append("Widen RSI range: use (rsi > 30) & (rsi < 70) instead of narrow ranges")
    elif results["total_trades"] < 20:
        issues.append(f"Too few trades ({results['total_trades']}) - need at least 30 for statistics")
        suggestions.append("Loosen entry conditions - widen RSI thresholds")
    elif results["total_trades"] > 300:
        issues.append(f"Too many trades ({results['total_trades']}) - overtrading")
        suggestions.append("Add one trend filter to reduce noise")
    elif results["total_profit_pct"] < 0:
        issues.append(f"Strategy losing money ({results['total_profit_pct']:.2f}%)")
        if results["win_rate"] < 40:
            suggestions.append(f"Low win rate ({results['win_rate']:.1f}%) - add trend confirmation (price > ema_21)")
        else:
            suggestions.append("Exits may be too early - raise RSI exit threshold to 75")
    else:
        issues.append(f"Strategy profitable at {results['total_profit_pct']:.2f}% - fine-tune for better performance")
        if results["win_rate"] < 50:
            suggestions.append("Add one more quality filter to improve win rate")

    if results["max_drawdown_pct"] > 20:
        issues.append(f"High drawdown ({results['max_drawdown_pct']:.2f}%)")
        suggestions.append("Consider tighter stoploss or trend filter")

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

    # Build system prompt - emphasize GENERATING TRADES as priority #1
    system_prompt = """You are an expert FreqTrade strategy developer. Your task is to generate a WORKING trading strategy.

ABSOLUTE PRIORITY #1: THE STRATEGY MUST GENERATE TRADES!
A strategy with 0 trades is USELESS. It's better to have 100 losing trades than 0 trades.

CODE REQUIREMENTS:
1. Return ONLY valid Python code - no explanations, no markdown, no ```
2. Class name MUST be 'FreqAIStrategy'
3. Every indicator used in entry/exit MUST be defined in populate_indicators()
4. Use ONLY: ta.RSI, ta.MACD, ta.EMA, ta.SMA, ta.ATR, ta.ADX, ta.BBANDS
5. BBANDS must use floats: ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)

ENTRY CONDITIONS - KEEP IT SIMPLE:
- Use 3-4 conditions MAX connected with &
- Use | (OR) to create multiple entry opportunities
- GOOD: (ema_8 > ema_21) & (rsi > 30) & (rsi < 70) & (volume > 0)
- BAD: More than 5 conditions = zero trades
- BAD: Using .shift() comparisons = often zero trades
- BAD: Narrow RSI ranges like (rsi > 45) & (rsi < 55) = zero trades

WORKING ENTRY EXAMPLE:
    dataframe.loc[
        (
            (dataframe["ema_8"] > dataframe["ema_21"]) &
            (dataframe["rsi"] > 30) & (dataframe["rsi"] < 70) &
            (dataframe["macd"] > dataframe["macdsignal"]) &
            (dataframe["volume"] > 0)
        ) |
        (
            (dataframe["rsi"] < 30) &
            (dataframe["close"] < dataframe["bb_lower"]) &
            (dataframe["volume"] > 0)
        ),
        "enter_long"
    ] = 1

WORKING EXIT EXAMPLE:
    dataframe.loc[
        (dataframe["rsi"] > 70) |
        (
            (dataframe["ema_8"] < dataframe["ema_21"]) &
            (dataframe["macd"] < dataframe["macdsignal"])
        ),
        "exit_long"
    ] = 1

DEFAULT SETTINGS:
    minimal_roi = {"0": 0.02, "30": 0.01, "60": 0.005}
    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02"""

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
TARGET: Generate 30-100 trades with positive profit over 60 days.

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

    # Copy initial strategy
    initial_strategy = STRATEGIES_DIR / "FreqAIStrategy.py"
    shutil.copy2(initial_strategy, USER_DATA_DIR / "strategies" / "FreqAIStrategy.py")

    print("=" * 60)
    print("FreqTrade AI Strategy Optimizer (Hyperopt + DeepSeek)")
    print("=" * 60)
    print(f"Timeframe: 5m")
    print(f"Target profit: {TARGET_PROFIT}%")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Hyperopt epochs per iteration: {HYPEROPT_EPOCHS}")
    print(f"AI Model: DeepSeek ({DEEPSEEK_MODEL})")
    print("=" * 60)

    # Load current strategy
    with open(USER_DATA_DIR / "strategies" / "FreqAIStrategy.py") as f:
        current_strategy = f.read()

    best_profit = float('-inf')
    best_strategy = current_strategy
    iteration_results = []
    consecutive_zero_trades = 0  # Track consecutive iterations with 0 trades

    for iteration in range(1, MAX_ITERATIONS + 1):
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
            best_profit = results["total_profit_pct"]
            best_strategy = current_strategy
            print(f"  *** New best: {best_profit:.2f}% ***")

        # Check target
        if results["total_profit_pct"] >= TARGET_PROFIT and results["total_trades"] >= 20:
            print(f"\n{'='*60}")
            print(f"TARGET ACHIEVED! Profit: {results['total_profit_pct']:.2f}%")
            print(f"{'='*60}")

            # Save winning strategy to backup directory
            backup_dir = USER_DATA_DIR / "strategy_backups"
            backup_dir.mkdir(exist_ok=True)
            with open(backup_dir / f"FreqAIStrategy_winning_{results['total_profit_pct']:.1f}pct.py", 'w') as f:
                f.write(current_strategy)
            break

        # Step 2: Run hyperopt ONLY if we have trades (otherwise it's useless)
        hyperopt_params = None
        if results["total_trades"] >= 20:
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
