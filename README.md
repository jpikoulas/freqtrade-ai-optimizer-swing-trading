# FreqTrade AI Strategy Optimizer - Swing Trading

Automated system that generates, backtests, and iteratively improves FreqTrade **swing trading** strategies using DeepSeek AI and hyperparameter optimization.

**Timeframe:** 1-hour (1h) - optimized for swing trades held for days to weeks.

## Features

- **Swing Trading Focus**: Optimized for 1h timeframe with longer EMAs (20, 50, 100) and trend-following strategies
- **AI-Powered Strategy Generation**: Uses DeepSeek API to analyze results and generate improved strategies
- **Hyperopt Integration**: Automatically optimizes ROI, stoploss, and trailing stop parameters
- **Fully Dockerized**: Runs entirely in containers - no Python installation required
- **Automated Optimization Loop**: Backtests, analyzes, improves, repeats
- **Smart Fallback**: Resets to base strategy after consecutive zero-trade iterations
- **Detailed Reporting**: Tracks all iterations and saves best strategies
- **Slack Notifications**: Get real-time alerts for new best strategies and target achievements
- **Crash Recovery**: State is saved after each iteration, automatically resumes on restart

## Quick Start

1. **Clone and setup:**
   ```bash
   cd freqtrade-ai-optimizer-swing-trading
   cp .env.example .env
   # Edit .env and add your DEEPSEEK_API_KEY
   ```

2. **Run the optimizer:**
   ```bash
   ./run_optimizer.sh
   ```

   Or with custom parameters:
   ```bash
   ./run_optimizer.sh --iterations 30 --days 90 --target 10
   ```

## Command Line Options

```
Usage: ./run_optimizer.sh [OPTIONS]

Options:
  -i, --iterations N    Maximum optimization iterations (default: 10)
  -t, --target N        Target profit percentage (default: 5.0)
  -d, --days N          Backtest period in days (default: 90)
  -e, --epochs N        Hyperopt epochs per iteration (default: 100)
  -m, --model NAME      DeepSeek model: deepseek-chat or deepseek-reasoner (default: deepseek-chat)
  -b, --background      Run in background (detached mode)
  -f, --forever         Run indefinitely until target profit is reached
  -h, --help            Show this help message

Management commands:
  --status              Show status of running optimizer
  --logs                Follow logs of running optimizer
  --stop                Stop running optimizer

Examples:
  ./run_optimizer.sh --iterations 30 --target 10
  ./run_optimizer.sh -i 20 -m deepseek-reasoner
  ./run_optimizer.sh --forever --background    # Run until 5% profit, in background
  ./run_optimizer.sh --target 15 --forever -b  # Run until 15% profit, in background
```

## Background Execution

For long-running optimization sessions, use background mode:

```bash
# Run indefinitely in background until 5% profit target
./run_optimizer.sh --forever --background

# Run with custom target (15% profit) in background
./run_optimizer.sh -f -b -t 15

# Check status
./run_optimizer.sh --status

# Follow live logs
./run_optimizer.sh --logs

# Or tail the log file directly
tail -f user_data/optimizer.log

# Stop the optimizer
./run_optimizer.sh --stop
```

The optimizer runs in a named Docker container (`freqtrade-swing-optimizer`) for easy management. Only one optimizer instance can run at a time.

## Slack Notifications

When configured with a Slack webhook URL, the optimizer sends notifications for:

- **Optimizer Started**: Configuration summary when optimization begins
- **New Best Strategy**: Real-time alert when a new best profit is achieved, including full results
- **Target Achieved**: Celebration notification when the target profit is reached

Example notification:
```
🎯 New Best SWING Strategy Found!
Time: 2025-01-21 14:32:15
Iteration: 5

Results:
  Total Trades: 42
  Total Profit: 6.78%
  Win Rate: 61.90%
  Max Drawdown: 5.23%
  Sharpe Ratio: 1.82
  Avg Profit/Trade: 1.61%
  *** New best: 6.78% ***

Previous best: 3.45%
```

To set up Slack notifications:
1. Create a Slack app at https://api.slack.com/apps
2. Enable Incoming Webhooks
3. Create a webhook for your channel
4. Add the webhook URL to your `.env` file

## Configuration

### Environment Variables

Create a `.env` file with:

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes | Your DeepSeek API key |
| `SLACK_WEBHOOK_URL` | No | Slack webhook URL for notifications |

Example `.env` file:
```bash
DEEPSEEK_API_KEY=your-deepseek-api-key-here

# Slack webhook URL for notifications (optional)
# Get your webhook URL from: https://api.slack.com/messaging/webhooks
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
```

### FreqTrade Config

Edit `config/freqtrade_config.json` to customize:
- Trading pairs (default: BTC/USDT)
- Exchange settings
- Stake amount and currency

## Project Structure

```
freqtrade-ai-optimizer-swing-trading/
├── run_optimizer.sh           # Main entry point - run this!
├── run_local.py               # Optimization loop logic
├── config/
│   └── freqtrade_config.json  # FreqTrade configuration
├── strategies/
│   └── FreqAIStrategy.py      # Initial strategy template
├── user_data/
│   ├── strategies/            # Active strategy (FreqAIStrategy.py)
│   ├── strategy_backups/      # Iteration backups and best strategies
│   ├── data/binance/          # Historical price data
│   ├── backtest_results/      # Backtest output files
│   ├── hyperopt_results/      # Hyperopt optimization results
│   ├── optimization_report.json  # Full iteration results
│   └── optimizer.log          # Background execution logs
└── .env                       # API keys (create this)
```

## How It Works

1. **Backtest**: Run strategy against historical data (default: 90 days of BTC/USDT 1h)
2. **Analyze**: Parse results (trades, profit, win rate, drawdown, Sharpe ratio)
3. **Check Target**: If profit >= target AND trades >= 10, save winning strategy and exit
4. **Hyperopt**: If trades >= 10, optimize ROI/stoploss/trailing parameters
5. **AI Improve**: Send results to DeepSeek to generate improved swing trading conditions
6. **Repeat**: Loop until target achieved or max iterations reached

### Swing Trading Strategy Characteristics

- **Timeframe**: 1h candles for longer-term trend analysis
- **EMAs**: Uses 20, 50, and 100 period EMAs for trend identification
- **Trade Frequency**: Expects 10-50 trades over 90 days (fewer but higher quality)
- **Profit Targets**: Larger ROI targets (5-15% per trade)
- **Hold Duration**: Trades held for days to weeks

### Zero-Trade Fallback

If the AI generates a strategy with 0 trades for 2 consecutive iterations, the optimizer automatically resets to a proven base strategy that reliably generates trades.

## Output

After running, find results in:

- `user_data/strategies/FreqAIStrategy.py` - Current active strategy
- `user_data/strategy_backups/FreqAIStrategy_winning_*.py` - Winning strategies
- `user_data/strategy_backups/FreqAIStrategy_iter*.py` - Each iteration's strategy
- `user_data/strategy_backups/FreqAIStrategy_best_*.py` - Best strategy from run
- `user_data/optimization_report.json` - Full iteration results with metrics

## Manual Commands

**Download more historical data:**
```bash
docker compose run --rm freqtrade download-data \
  --config /freqtrade/config/freqtrade_config.json \
  --timerange 20240101-20260118 \
  --timeframe 1h
```

**Run single backtest:**
```bash
docker compose run --rm freqtrade backtesting \
  --config /freqtrade/config/freqtrade_config.json \
  --strategy FreqAIStrategy \
  --timerange 20251020-20260118
```

**Check available data:**
```bash
docker compose run --rm freqtrade list-data \
  --config /freqtrade/config/freqtrade_config.json \
  --show-timerange
```

**Force rebuild optimizer image:**
```bash
rm -f .optimizer_image_built
./run_optimizer.sh
```

**Build freqtrade image:**
```bash
docker compose build freqtrade
```

## Performance Notes

- 1-hour timeframe with 90 days = ~2,160 candles per backtest
- Backtests run faster than scalping due to fewer candles
- Each iteration takes 1-3 minutes (backtest + hyperopt + AI generation)
- Full optimization (30 iterations) typically takes 30-60 minutes
- DeepSeek API calls are fast (~2-5 seconds per generation)

## Troubleshooting

**Zero trades generated:**
- The optimizer will auto-reset after 2 consecutive zero-trade iterations
- Swing trading expects fewer trades (10-50 over 90 days) - this is normal
- Check if entry conditions are too strict
- Verify historical data is available for the timerange

**Docker errors:**
- Ensure Docker is running: `docker info`
- Check Docker socket permissions: `ls -la /var/run/docker.sock`

**API errors:**
- Verify DEEPSEEK_API_KEY is set correctly in `.env`
- Check DeepSeek API status

**Slack notifications not working:**
- Verify SLACK_WEBHOOK_URL is correctly set in `.env`
- Test the webhook manually with curl:
  ```bash
  curl -X POST -H 'Content-type: application/json' \
    --data '{"text":"Test message"}' \
    YOUR_WEBHOOK_URL
  ```

## License

MIT
