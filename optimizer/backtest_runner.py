"""
Run FreqTrade backtests via Docker.
"""
import json
import logging
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from result_parser import BacktestResult, find_latest_results, parse_backtest_results

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Runs FreqTrade backtests via Docker."""

    def __init__(
        self,
        user_data_dir: Path,
        config_path: Path,
        strategy_name: str = "FreqAIStrategy",
        backtest_days: int = 60,
    ):
        self.user_data_dir = user_data_dir
        self.config_path = config_path
        self.strategy_name = strategy_name
        self.backtest_days = backtest_days
        self.results_dir = user_data_dir / "backtest_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_timerange(self) -> str:
        """Calculate timerange for backtesting (last N days)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.backtest_days)
        return f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

    def copy_strategy_to_user_data(self, strategy_path: Path) -> bool:
        """
        Copy strategy file to user_data/strategies directory.

        Args:
            strategy_path: Path to the strategy file to copy

        Returns:
            True if successful, False otherwise
        """
        try:
            dest_dir = self.user_data_dir / "strategies"
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{self.strategy_name}.py"
            shutil.copy2(strategy_path, dest_path)
            logger.info(f"Copied strategy to {dest_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy strategy: {e}")
            return False

    def run_backtest(self, timeout: int = 1800) -> Tuple[bool, Optional[str]]:
        """
        Run backtest using docker-compose.

        Args:
            timeout: Maximum time to wait for backtest (seconds)

        Returns:
            Tuple of (success, error_message)
        """
        timerange = self.get_timerange()

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.user_data_dir.absolute()}:/freqtrade/user_data",
            "-v", f"{self.config_path.parent.absolute()}:/freqtrade/config",
            "freqtrade-ai-optimizer-freqtrade:latest",
            "backtesting",
            "--config", "/freqtrade/config/freqtrade_config.json",
            "--strategy", self.strategy_name,
            "--timerange", timerange,
            "--export", "trades",
            "--export-filename", f"/freqtrade/user_data/backtest_results/backtest-result-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ]

        logger.info(f"Running backtest command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                logger.error(f"Backtest failed: {error_msg}")
                return False, error_msg

            logger.info("Backtest completed successfully")
            return True, None

        except subprocess.TimeoutExpired:
            error_msg = f"Backtest timed out after {timeout} seconds"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error running backtest: {e}"
            logger.error(error_msg)
            return False, error_msg

    def get_results(self) -> Optional[BacktestResult]:
        """
        Get the latest backtest results.

        Returns:
            BacktestResult object or None if no results found
        """
        results_path = find_latest_results(self.results_dir)

        if results_path is None:
            return None

        return parse_backtest_results(results_path)

    def run_and_get_results(
        self, strategy_path: Path, timeout: int = 1800
    ) -> Tuple[Optional[BacktestResult], Optional[str]]:
        """
        Run backtest and return results.

        Args:
            strategy_path: Path to the strategy file
            timeout: Maximum time to wait for backtest

        Returns:
            Tuple of (BacktestResult or None, error_message or None)
        """
        # Copy strategy to user_data
        if not self.copy_strategy_to_user_data(strategy_path):
            return None, "Failed to copy strategy file"

        # Run backtest
        success, error = self.run_backtest(timeout)

        if not success:
            return None, error

        # Parse results
        results = self.get_results()

        if results is None:
            return None, "Failed to parse backtest results"

        return results, None


def download_data(
    user_data_dir: Path,
    config_path: Path,
    pairs: list[str] = None,
    timeframes: list[str] = None,
    days: int = 60,
) -> bool:
    """
    Download historical data for backtesting.

    Args:
        user_data_dir: Path to user_data directory
        config_path: Path to config file
        pairs: List of pairs to download (default: BTC/USDT, ETH/USDT)
        timeframes: List of timeframes (default: 1m, 5m, 15m)
        days: Number of days to download

    Returns:
        True if successful, False otherwise
    """
    if pairs is None:
        pairs = ["BTC/USDT", "ETH/USDT"]
    if timeframes is None:
        timeframes = ["1m", "5m", "15m"]

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{user_data_dir.absolute()}:/freqtrade/user_data",
        "-v", f"{config_path.parent.absolute()}:/freqtrade/config",
        "freqtrade-ai-optimizer-freqtrade:latest",
        "download-data",
        "--config", "/freqtrade/config/freqtrade_config.json",
        "--pairs", *pairs,
        "--timeframe", *timeframes,
        "--days", str(days)
    ]

    logger.info(f"Downloading data: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for download
        )

        if result.returncode != 0:
            logger.error(f"Data download failed: {result.stderr}")
            return False

        logger.info("Data download completed successfully")
        return True

    except subprocess.TimeoutExpired:
        logger.error("Data download timed out")
        return False
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False
