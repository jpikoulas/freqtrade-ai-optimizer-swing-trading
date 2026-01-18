#!/usr/bin/env python3
"""
FreqTrade AI Strategy Optimizer

Main optimization loop that:
1. Runs backtests on current strategy
2. Parses results
3. Uses Claude to generate improved strategies
4. Repeats until target profit is achieved or max iterations reached
"""
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from backtest_runner import BacktestRunner, download_data
from result_parser import BacktestResult
from strategy_generator import StrategyGenerator, load_strategy, save_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/user_data/optimizer.log')
    ]
)
logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Main optimization loop controller."""

    def __init__(
        self,
        anthropic_api_key: str,
        max_iterations: int = 25,
        target_profit: float = 5.0,
        backtest_days: int = 60,
        user_data_dir: Path = Path("/app/user_data"),
        config_path: Path = Path("/app/config/freqtrade_config.json"),
        initial_strategy_path: Path = Path("/app/strategies/FreqAIStrategy.py"),
    ):
        self.max_iterations = max_iterations
        self.target_profit = target_profit
        self.backtest_days = backtest_days
        self.user_data_dir = user_data_dir
        self.config_path = config_path
        self.initial_strategy_path = initial_strategy_path

        # Initialize components
        self.backtest_runner = BacktestRunner(
            user_data_dir=user_data_dir,
            config_path=config_path,
            backtest_days=backtest_days,
        )
        self.strategy_generator = StrategyGenerator(api_key=anthropic_api_key)

        # Tracking
        self.iteration_results: list[dict] = []
        self.best_result: Optional[BacktestResult] = None
        self.best_strategy: Optional[str] = None

    def run(self) -> bool:
        """
        Run the optimization loop.

        Returns:
            True if target profit achieved, False otherwise
        """
        logger.info("=" * 60)
        logger.info("FreqTrade AI Strategy Optimizer")
        logger.info("=" * 60)
        logger.info(f"Target profit: {self.target_profit}%")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Backtest period: {self.backtest_days} days")
        logger.info("=" * 60)

        # Load initial strategy
        current_strategy = load_strategy(self.initial_strategy_path)
        if current_strategy is None:
            logger.error("Failed to load initial strategy")
            return False

        current_strategy_path = self.initial_strategy_path

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}")

            # Run backtest
            logger.info("Running backtest...")
            result, error = self.backtest_runner.run_and_get_results(
                strategy_path=current_strategy_path
            )

            if result is None:
                logger.error(f"Backtest failed: {error}")
                # Save the failing strategy for debugging
                self._save_iteration_strategy(current_strategy, iteration, failed=True)

                # Try to generate a fixed strategy
                logger.info("Attempting to generate a new strategy...")
                new_strategy = self._generate_fallback_strategy(current_strategy, error)

                if new_strategy is None:
                    logger.error("Could not generate fallback strategy, continuing with original")
                    continue

                current_strategy = new_strategy
                current_strategy_path = self._save_iteration_strategy(
                    current_strategy, iteration
                )
                continue

            # Log results
            logger.info(result.get_summary())

            # Track results
            self._track_result(iteration, result, current_strategy)

            # Check if target achieved
            if result.meets_target(self.target_profit):
                logger.info("=" * 60)
                logger.info(f"TARGET ACHIEVED! Profit: {result.total_profit_pct:.2f}%")
                logger.info("=" * 60)
                self._save_winning_strategy(current_strategy, result)
                self._generate_report()
                return True

            # Update best result
            if self.best_result is None or result.total_profit_pct > self.best_result.total_profit_pct:
                self.best_result = result
                self.best_strategy = current_strategy
                logger.info(f"New best result: {result.total_profit_pct:.2f}%")

            # Generate improved strategy
            logger.info("Generating improved strategy with Claude...")
            improved_strategy = self.strategy_generator.generate_improved_strategy(
                current_strategy=current_strategy,
                backtest_result=result,
                iteration=iteration,
            )

            if improved_strategy is None:
                logger.error("Failed to generate improved strategy")
                continue

            # Save and update current strategy
            current_strategy = improved_strategy
            current_strategy_path = self._save_iteration_strategy(
                current_strategy, iteration
            )

        # Max iterations reached
        logger.info("=" * 60)
        logger.info("MAX ITERATIONS REACHED")
        logger.info("=" * 60)

        if self.best_result:
            logger.info(f"Best result achieved: {self.best_result.total_profit_pct:.2f}%")
            self._save_winning_strategy(self.best_strategy, self.best_result, is_best=True)

        self._generate_report()
        return False

    def _generate_fallback_strategy(
        self, current_strategy: str, error: Optional[str]
    ) -> Optional[str]:
        """Generate a fallback strategy when backtest fails."""
        # Create a mock result for the generator
        mock_result = BacktestResult(
            total_profit_pct=-10.0,
            total_profit_abs=0,
            win_rate=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            max_drawdown_pct=0,
            avg_profit_pct=0,
            avg_duration="N/A",
            best_pair="N/A",
            worst_pair="N/A",
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            profit_factor=0,
            expectancy=0,
            raw_data={"error": error or "Unknown error"}
        )

        return self.strategy_generator.generate_improved_strategy(
            current_strategy=current_strategy,
            backtest_result=mock_result,
            iteration=0,
        )

    def _track_result(
        self, iteration: int, result: BacktestResult, strategy: str
    ) -> None:
        """Track iteration results."""
        self.iteration_results.append({
            "iteration": iteration,
            "profit_pct": result.total_profit_pct,
            "win_rate": result.win_rate,
            "total_trades": result.total_trades,
            "max_drawdown": result.max_drawdown_pct,
            "sharpe_ratio": result.sharpe_ratio,
            "timestamp": datetime.now().isoformat(),
        })

    def _save_iteration_strategy(
        self, strategy: str, iteration: int, failed: bool = False
    ) -> Path:
        """Save strategy for an iteration."""
        suffix = "_failed" if failed else ""
        filename = f"FreqAIStrategy_iter{iteration:02d}{suffix}.py"
        output_path = self.user_data_dir / "strategies" / filename
        save_strategy(strategy, output_path)
        return output_path

    def _save_winning_strategy(
        self, strategy: str, result: BacktestResult, is_best: bool = False
    ) -> None:
        """Save the winning or best strategy."""
        suffix = "best" if is_best else "winning"
        filename = f"FreqAIStrategy_{suffix}_{result.total_profit_pct:.1f}pct.py"
        output_path = self.user_data_dir / "strategies" / filename
        save_strategy(strategy, output_path)

        # Also save as the main strategy
        main_path = self.user_data_dir / "strategies" / "FreqAIStrategy.py"
        save_strategy(strategy, main_path)

    def _generate_report(self) -> None:
        """Generate optimization report."""
        report = {
            "summary": {
                "target_profit": self.target_profit,
                "max_iterations": self.max_iterations,
                "iterations_run": len(self.iteration_results),
                "target_achieved": (
                    self.best_result is not None and
                    self.best_result.total_profit_pct >= self.target_profit
                ),
                "best_profit": (
                    self.best_result.total_profit_pct if self.best_result else None
                ),
            },
            "iterations": self.iteration_results,
        }

        report_path = self.user_data_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {report_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Iterations run: {len(self.iteration_results)}")

        if self.iteration_results:
            profits = [r["profit_pct"] for r in self.iteration_results]
            logger.info(f"Profit range: {min(profits):.2f}% to {max(profits):.2f}%")
            logger.info(f"Best profit: {max(profits):.2f}%")

        logger.info("=" * 60)


def main():
    """Main entry point."""
    # Get configuration from environment
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        logger.error("ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    max_iterations = int(os.environ.get("MAX_ITERATIONS", "25"))
    target_profit = float(os.environ.get("TARGET_PROFIT", "5.0"))
    backtest_days = int(os.environ.get("BACKTEST_DAYS", "60"))

    # Setup paths
    user_data_dir = Path("/app/user_data")
    config_path = Path("/app/config/freqtrade_config.json")
    initial_strategy_path = Path("/app/strategies/FreqAIStrategy.py")

    # Ensure directories exist
    (user_data_dir / "strategies").mkdir(parents=True, exist_ok=True)
    (user_data_dir / "data").mkdir(parents=True, exist_ok=True)
    (user_data_dir / "backtest_results").mkdir(parents=True, exist_ok=True)

    # Copy initial strategy
    shutil.copy2(
        initial_strategy_path,
        user_data_dir / "strategies" / "FreqAIStrategy.py"
    )

    # Check if data exists, download if not
    data_dir = user_data_dir / "data" / "binance"
    if not data_dir.exists() or not list(data_dir.glob("*.json")):
        logger.info("Downloading historical data...")
        if not download_data(user_data_dir, config_path, days=backtest_days):
            logger.error("Failed to download data")
            sys.exit(1)

    # Run optimizer
    optimizer = StrategyOptimizer(
        anthropic_api_key=anthropic_api_key,
        max_iterations=max_iterations,
        target_profit=target_profit,
        backtest_days=backtest_days,
        user_data_dir=user_data_dir,
        config_path=config_path,
        initial_strategy_path=user_data_dir / "strategies" / "FreqAIStrategy.py",
    )

    success = optimizer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
