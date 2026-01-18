"""
Parse FreqTrade backtest results from JSON output.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for parsed backtest results."""
    total_profit_pct: float
    total_profit_abs: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_drawdown_pct: float
    avg_profit_pct: float
    avg_duration: str
    best_pair: str
    worst_pair: str
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    expectancy: float
    raw_data: dict

    def meets_target(self, target_profit: float = 5.0) -> bool:
        """Check if results meet the target profit."""
        return self.total_profit_pct >= target_profit

    def get_summary(self) -> str:
        """Get a human-readable summary of results."""
        return f"""
Backtest Results Summary:
========================
Total Profit: {self.total_profit_pct:.2f}%
Win Rate: {self.win_rate:.2f}%
Total Trades: {self.total_trades}
Max Drawdown: {self.max_drawdown_pct:.2f}%
Avg Profit/Trade: {self.avg_profit_pct:.2f}%
Avg Duration: {self.avg_duration}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Profit Factor: {self.profit_factor:.2f}
"""

    def get_analysis(self) -> str:
        """Get analysis for Claude to improve the strategy."""
        issues = []
        strengths = []

        # Analyze profit
        if self.total_profit_pct < 0:
            issues.append(f"Strategy is losing money ({self.total_profit_pct:.2f}%)")
        elif self.total_profit_pct < 2:
            issues.append(f"Profit is below target ({self.total_profit_pct:.2f}% vs 5% target)")
        else:
            strengths.append(f"Profitable at {self.total_profit_pct:.2f}%")

        # Analyze win rate
        if self.win_rate < 40:
            issues.append(f"Low win rate ({self.win_rate:.2f}%) - entry signals may be poor")
        elif self.win_rate > 60:
            strengths.append(f"Good win rate ({self.win_rate:.2f}%)")

        # Analyze drawdown
        if self.max_drawdown_pct > 20:
            issues.append(f"High drawdown ({self.max_drawdown_pct:.2f}%) - risk management needs improvement")
        elif self.max_drawdown_pct < 10:
            strengths.append(f"Controlled drawdown ({self.max_drawdown_pct:.2f}%)")

        # Analyze trade count
        if self.total_trades < 10:
            issues.append(f"Too few trades ({self.total_trades}) - entry conditions may be too strict")
        elif self.total_trades > 1000:
            issues.append(f"Too many trades ({self.total_trades}) - may be overtrading")
        else:
            strengths.append(f"Reasonable trade frequency ({self.total_trades} trades)")

        # Analyze profit factor
        if self.profit_factor < 1.0:
            issues.append(f"Profit factor below 1 ({self.profit_factor:.2f}) - losses exceed gains")
        elif self.profit_factor > 1.5:
            strengths.append(f"Good profit factor ({self.profit_factor:.2f})")

        # Analyze Sharpe ratio
        if self.sharpe_ratio < 0:
            issues.append(f"Negative Sharpe ratio ({self.sharpe_ratio:.2f}) - poor risk-adjusted returns")
        elif self.sharpe_ratio > 1:
            strengths.append(f"Good Sharpe ratio ({self.sharpe_ratio:.2f})")

        analysis = "STRENGTHS:\n"
        if strengths:
            analysis += "\n".join(f"- {s}" for s in strengths)
        else:
            analysis += "- None identified"

        analysis += "\n\nISSUES TO ADDRESS:\n"
        if issues:
            analysis += "\n".join(f"- {i}" for i in issues)
        else:
            analysis += "- None identified"

        return analysis


def parse_backtest_results(results_path: Path) -> Optional[BacktestResult]:
    """
    Parse backtest results from FreqTrade JSON output.

    Args:
        results_path: Path to the backtest results JSON file

    Returns:
        BacktestResult object or None if parsing fails
    """
    try:
        with open(results_path) as f:
            data = json.load(f)

        # FreqTrade stores results under 'strategy' key with strategy name
        strategy_results = None
        if "strategy" in data:
            strategy_results = data["strategy"]
        elif "strategy_comparison" in data:
            # Get the first strategy results
            for strategy_name, results in data.get("strategy", {}).items():
                strategy_results = results
                break

        # Try to find results in different formats
        if strategy_results is None:
            # Check if it's directly in the root
            if "total_profit_pct" in data or "profit_total" in data:
                strategy_results = data
            else:
                # Look for any strategy key
                for key in data:
                    if isinstance(data[key], dict) and ("total_profit" in str(data[key]) or "trades" in str(data[key])):
                        strategy_results = data[key]
                        break

        if strategy_results is None:
            logger.error(f"Could not find strategy results in {results_path}")
            return None

        # Extract metrics with fallbacks
        total_profit_pct = strategy_results.get("profit_total", 0) * 100
        if "profit_total_pct" in strategy_results:
            total_profit_pct = strategy_results["profit_total_pct"]

        total_trades = strategy_results.get("total_trades", 0)
        winning_trades = strategy_results.get("wins", 0)
        losing_trades = strategy_results.get("losses", 0)

        win_rate = 0
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100

        return BacktestResult(
            total_profit_pct=total_profit_pct,
            total_profit_abs=strategy_results.get("profit_total_abs", 0),
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown_pct=abs(strategy_results.get("max_drawdown", 0) * 100),
            avg_profit_pct=strategy_results.get("profit_mean", 0) * 100,
            avg_duration=strategy_results.get("holding_avg", "N/A"),
            best_pair=strategy_results.get("best_pair", "N/A"),
            worst_pair=strategy_results.get("worst_pair", "N/A"),
            sharpe_ratio=strategy_results.get("sharpe", 0),
            sortino_ratio=strategy_results.get("sortino", 0),
            calmar_ratio=strategy_results.get("calmar", 0),
            profit_factor=strategy_results.get("profit_factor", 0),
            expectancy=strategy_results.get("expectancy", 0),
            raw_data=strategy_results
        )

    except FileNotFoundError:
        logger.error(f"Results file not found: {results_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {results_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error parsing backtest results: {e}")
        return None


def find_latest_results(results_dir: Path) -> Optional[Path]:
    """
    Find the most recent backtest results file.

    Args:
        results_dir: Directory containing backtest results

    Returns:
        Path to the latest results file or None
    """
    results_files = list(results_dir.glob("backtest-result-*.json"))

    if not results_files:
        logger.warning(f"No backtest results found in {results_dir}")
        return None

    # Sort by modification time, newest first
    results_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    return results_files[0]
