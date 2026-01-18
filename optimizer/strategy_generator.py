"""
Generate improved strategies using Claude API.
"""
import ast
import logging
import re
from pathlib import Path
from typing import Optional

import anthropic

from result_parser import BacktestResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert quantitative trader and Python developer specializing in FreqTrade and FreqAI strategies. Your task is to improve trading strategies based on backtest results.

IMPORTANT RULES:
1. Return ONLY valid Python code - no explanations, no markdown, no code blocks
2. The strategy must be a valid FreqTrade IStrategy class with FreqAI integration
3. Keep the class name as 'FreqAIStrategy'
4. Maintain all required FreqAI methods (feature_engineering_*, set_freqai_targets, etc.)
5. Make targeted improvements based on the analysis provided
6. Do not remove essential FreqAI functionality
7. Ensure all imports are present at the top of the file
8. Test any new indicators/features for logical consistency

FOCUS AREAS FOR IMPROVEMENT:
- Feature engineering: Add or modify technical indicators
- Entry/exit conditions: Improve signal quality
- Risk management: Adjust stoploss, ROI, trailing stop
- Model parameters: Fine-tune FreqAI configuration
- Time-based filters: Add market session or volatility filters"""

IMPROVEMENT_PROMPT = """Here is the current FreqAI trading strategy:

```python
{strategy_code}
```

BACKTEST RESULTS (Last 2 months on BTC/USDT 1-minute):
- Total Profit: {profit:.2f}%
- Win Rate: {win_rate:.2f}%
- Total Trades: {trades}
- Max Drawdown: {drawdown:.2f}%
- Average Profit per Trade: {avg_profit:.2f}%
- Sharpe Ratio: {sharpe:.2f}
- Profit Factor: {profit_factor:.2f}

ANALYSIS:
{analysis}

TARGET: We need to achieve at least 5% profit over 2 months.

Based on this analysis, generate an IMPROVED version of the strategy. Focus on fixing the identified issues while preserving what works well.

{specific_guidance}

Return ONLY the complete Python strategy code. No explanations, no markdown formatting."""


def get_specific_guidance(result: BacktestResult) -> str:
    """Generate specific improvement guidance based on results."""
    guidance = []

    if result.total_profit_pct < 0:
        guidance.append(
            "CRITICAL: Strategy is losing money. Consider:\n"
            "- Tightening entry conditions to reduce false signals\n"
            "- Adding trend filters (only trade in direction of trend)\n"
            "- Improving stop loss placement"
        )

    if result.win_rate < 40:
        guidance.append(
            "LOW WIN RATE: Consider:\n"
            "- Adding confirmation indicators for entries\n"
            "- Using stricter DI_threshold in FreqAI config\n"
            "- Adding volume or volatility filters"
        )

    if result.max_drawdown_pct > 15:
        guidance.append(
            "HIGH DRAWDOWN: Consider:\n"
            "- Reducing position size or max_open_trades\n"
            "- Tightening stoploss\n"
            "- Adding market regime filter to avoid trading in high volatility"
        )

    if result.total_trades < 20:
        guidance.append(
            "TOO FEW TRADES: Consider:\n"
            "- Relaxing entry conditions slightly\n"
            "- Reducing DI_threshold\n"
            "- Shortening label_period_candles for more frequent signals"
        )

    if result.total_trades > 500:
        guidance.append(
            "OVERTRADING: Consider:\n"
            "- Adding minimum profit target before entering\n"
            "- Adding cooldown period between trades\n"
            "- Strengthening entry confirmation signals"
        )

    if result.profit_factor < 1.0:
        guidance.append(
            "PROFIT FACTOR < 1: Consider:\n"
            "- Improving exit timing (earlier exits on winners, cut losers faster)\n"
            "- Better trailing stop configuration\n"
            "- Adding profit target exits"
        )

    if not guidance:
        guidance.append(
            "Strategy is performing reasonably. Focus on:\n"
            "- Fine-tuning existing parameters\n"
            "- Adding complementary indicators\n"
            "- Optimizing ROI table and stoploss"
        )

    return "\n\n".join(guidance)


class StrategyGenerator:
    """Generate improved strategies using Claude API."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate_improved_strategy(
        self,
        current_strategy: str,
        backtest_result: BacktestResult,
        iteration: int = 1,
    ) -> Optional[str]:
        """
        Generate an improved strategy based on backtest results.

        Args:
            current_strategy: Current strategy code
            backtest_result: Results from backtesting
            iteration: Current iteration number

        Returns:
            Improved strategy code or None if generation fails
        """
        specific_guidance = get_specific_guidance(backtest_result)

        prompt = IMPROVEMENT_PROMPT.format(
            strategy_code=current_strategy,
            profit=backtest_result.total_profit_pct,
            win_rate=backtest_result.win_rate,
            trades=backtest_result.total_trades,
            drawdown=backtest_result.max_drawdown_pct,
            avg_profit=backtest_result.avg_profit_pct,
            sharpe=backtest_result.sharpe_ratio,
            profit_factor=backtest_result.profit_factor,
            analysis=backtest_result.get_analysis(),
            specific_guidance=specific_guidance,
        )

        logger.info(f"Generating improved strategy (iteration {iteration})...")

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                system=SYSTEM_PROMPT,
            )

            response_text = message.content[0].text

            # Clean up the response
            improved_strategy = self._clean_response(response_text)

            # Validate the strategy
            if not self._validate_strategy(improved_strategy):
                logger.error("Generated strategy failed validation")
                return None

            logger.info("Successfully generated improved strategy")
            return improved_strategy

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return None

    def _clean_response(self, response: str) -> str:
        """Clean up Claude's response to extract pure Python code."""
        # Remove markdown code blocks if present
        response = re.sub(r'^```python\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^```\s*$', '', response, flags=re.MULTILINE)
        response = re.sub(r'```$', '', response)

        # Remove any leading/trailing whitespace
        response = response.strip()

        # Ensure it starts with imports or docstring
        if not (response.startswith('"""') or
                response.startswith("'''") or
                response.startswith("import") or
                response.startswith("from") or
                response.startswith("#")):
            # Try to find where the code actually starts
            lines = response.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if (line.strip().startswith('"""') or
                    line.strip().startswith("'''") or
                    line.strip().startswith("import") or
                    line.strip().startswith("from") or
                    line.strip().startswith("#")):
                    start_idx = i
                    break
            response = '\n'.join(lines[start_idx:])

        return response

    def _validate_strategy(self, strategy_code: str) -> bool:
        """
        Validate that the strategy code is valid Python and has required components.

        Args:
            strategy_code: Strategy code to validate

        Returns:
            True if valid, False otherwise
        """
        # Check Python syntax
        try:
            ast.parse(strategy_code)
        except SyntaxError as e:
            logger.error(f"Strategy has syntax error: {e}")
            return False

        # Check for required class
        if "class FreqAIStrategy" not in strategy_code:
            logger.error("Strategy missing 'class FreqAIStrategy'")
            return False

        # Check for required methods
        required_methods = [
            "populate_indicators",
            "populate_entry_trend",
            "populate_exit_trend",
        ]

        for method in required_methods:
            if f"def {method}" not in strategy_code:
                logger.error(f"Strategy missing required method: {method}")
                return False

        # Check for FreqAI feature engineering (at least one)
        freqai_methods = [
            "feature_engineering_expand_all",
            "feature_engineering_expand_basic",
            "feature_engineering_standard",
            "set_freqai_targets",
        ]

        has_freqai_method = any(
            f"def {method}" in strategy_code for method in freqai_methods
        )

        if not has_freqai_method:
            logger.warning("Strategy missing FreqAI feature engineering methods")
            # Don't fail, as the strategy might still work

        # Check for IStrategy import
        if "IStrategy" not in strategy_code:
            logger.error("Strategy missing IStrategy import")
            return False

        return True


def load_strategy(strategy_path: Path) -> Optional[str]:
    """Load strategy code from file."""
    try:
        with open(strategy_path) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load strategy from {strategy_path}: {e}")
        return None


def save_strategy(strategy_code: str, output_path: Path) -> bool:
    """Save strategy code to file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(strategy_code)
        logger.info(f"Saved strategy to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save strategy to {output_path}: {e}")
        return False
