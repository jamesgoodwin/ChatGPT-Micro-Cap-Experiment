#!/usr/bin/env python3
"""Unit tests for portfolio health check functionality."""

import unittest
import pandas as pd
import tempfile
import os
from datetime import datetime, timedelta
from trading_script import generate_chatgpt_prompt, generate_portfolio_summary, generate_market_data


class TestPortfolioHealth(unittest.TestCase):
    """Test portfolio health check functionality."""

    def setUp(self):
        """Set up test data."""
        # Create sample portfolio data
        self.sample_portfolio_data = [
            {
                'Date': '2025-08-18',
                'Ticker': 'ABEO',
                'Shares': 1,
                'Buy Price': '5.87',
                'Cost Basis': '5.87',
                'Stop Loss': '4.50',
                'Current Price': '6.20',
                'Total Value': '6.20',
                'PnL': '0.33',
                'Action': 'HOLD',
                'Cash Balance': '94.13',
                'Total Equity': '100.33'
            },
            {
                'Date': '2025-08-18',
                'Ticker': 'TOTAL',
                'Shares': '',
                'Buy Price': '',
                'Cost Basis': '',
                'Stop Loss': '',
                'Current Price': '',
                'Total Value': '100.33',
                'PnL': '0.33',
                'Action': 'SUMMARY',
                'Cash Balance': '94.13',
                'Total Equity': '100.33'
            }
        ]
        
        # Create temporary CSV file
        self.temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df = pd.DataFrame(self.sample_portfolio_data)
        df.to_csv(self.temp_csv.name, index=False)
        self.temp_csv.close()

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.temp_csv.name):
            os.unlink(self.temp_csv.name)

    def test_portfolio_health_check_with_string_values(self):
        """Test that portfolio health check handles string values correctly."""
        def check_portfolio_health(csv_path):
            """Portfolio health check function (same as in workflow)."""
            if not os.path.exists(csv_path):
                print(f'❌ Portfolio file not found: {csv_path}')
                return False
                
            df = pd.read_csv(csv_path)
            if df.empty:
                print(f'⚠️  Portfolio is empty: {csv_path}')
                return True
                
            # Check for recent data
            df['Date'] = pd.to_datetime(df['Date'])
            latest_date = df['Date'].max()
            days_old = (datetime.now() - latest_date).days
            
            print(f'📊 Portfolio: {csv_path}')
            print(f'   Latest update: {latest_date.strftime("%Y-%m-%d")} ({days_old} days ago)')
            
            if days_old > 7:
                print(f'⚠️  Data is {days_old} days old')
            elif days_old > 3:
                print(f'⚠️  Data is {days_old} days old')
            else:
                print(f'✅ Data is recent')
                
            # Check for TOTAL row
            total_rows = df[df['Ticker'] == 'TOTAL']
            if total_rows.empty:
                print('❌ No TOTAL summary row found')
                return False
            else:
                latest_total = total_rows.iloc[-1]
                # Convert to float to handle string values from CSV
                equity = float(latest_total['Total Equity'])
                cash = float(latest_total['Cash Balance'])
                print(f'   Total Equity: ${equity:.2f}')
                print(f'   Cash Balance: ${cash:.2f}')
                
            return True

        # Test the function - this should not raise any exceptions
        try:
            result = check_portfolio_health(self.temp_csv.name)
            self.assertTrue(result, "Portfolio health check should pass with valid data")
            print("✅ Portfolio health check passed with string values")
        except ValueError as e:
            self.fail(f"Portfolio health check failed with ValueError: {e}")
        except Exception as e:
            self.fail(f"Portfolio health check failed with unexpected error: {e}")

    def test_generate_portfolio_summary_with_string_values(self):
        """Test that portfolio summary generation handles string values correctly."""
        # Create portfolio DataFrame with string values (like from CSV)
        portfolio_data = [
            {
                'ticker': 'ABEO',
                'shares': 1,
                'buy_price': '5.87',  # String value
                'cost_basis': '5.87',  # String value
                'stop_loss': '4.50'    # String value
            }
        ]
        portfolio = pd.DataFrame(portfolio_data)
        cash = 94.13

        # Test that this doesn't raise formatting errors
        try:
            summary = generate_portfolio_summary(portfolio, cash)
            self.assertIsInstance(summary, str)
            self.assertGreater(len(summary), 0)
            print("✅ Portfolio summary generation passed with string values")
        except ValueError as e:
            self.fail(f"Portfolio summary generation failed with ValueError: {e}")
        except Exception as e:
            self.fail(f"Portfolio summary generation failed with unexpected error: {e}")

    def test_generate_chatgpt_prompt_with_string_values(self):
        """Test that ChatGPT prompt generation handles string values correctly."""
        # Create portfolio DataFrame with string values
        portfolio_data = [
            {
                'ticker': 'ABEO',
                'shares': 1,
                'buy_price': '5.87',  # String value
                'cost_basis': '5.87',  # String value
                'stop_loss': '4.50'    # String value
            }
        ]
        portfolio = pd.DataFrame(portfolio_data)
        cash = 94.13

        # Test that this doesn't raise formatting errors
        try:
            prompt = generate_chatgpt_prompt(portfolio, cash)
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 0)
            print("✅ ChatGPT prompt generation passed with string values")
        except ValueError as e:
            self.fail(f"ChatGPT prompt generation failed with ValueError: {e}")
        except Exception as e:
            self.fail(f"ChatGPT prompt generation failed with unexpected error: {e}")

    def test_generate_market_data(self):
        """Test that market data generation works correctly."""
        try:
            market_data = generate_market_data()
            self.assertIsInstance(market_data, str)
            self.assertGreater(len(market_data), 0)
            print("✅ Market data generation passed")
        except Exception as e:
            self.fail(f"Market data generation failed with error: {e}")


if __name__ == '__main__':
    print("🧪 Running portfolio health unit tests...")
    unittest.main(verbosity=2)
