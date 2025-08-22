#!/usr/bin/env python3
"""Unit tests for robust price fetching helper."""

import unittest
from unittest.mock import patch
import pandas as pd
from trading_script import fetch_intraday_or_last_close


class TestPriceFetchHelper(unittest.TestCase):
	def test_intraday_available(self):
		# Create intraday-like DataFrame
		intraday = pd.DataFrame({
			"Close": [10.0, 10.5, 10.7],
			"Low": [9.9, 10.1, 10.6],
		})

		with patch("trading_script.yf.download") as mock_dl:
			# First call: intraday, Second call shouldn't be reached
			mock_dl.return_value = intraday
			price, low = fetch_intraday_or_last_close("TEST")
			self.assertEqual(price, 10.7)
			self.assertEqual(low, 9.9)

	def test_daily_fallback(self):
		# Empty intraday, valid daily fallback
		empty = pd.DataFrame()
		daily = pd.DataFrame({
			"Close": [10.0, 10.2],
			"Low": [9.8, 10.1],
		})

		with patch("trading_script.yf.download") as mock_dl:
			# Sequence: intraday empty, daily valid
			mock_dl.side_effect = [empty, daily]
			price, low = fetch_intraday_or_last_close("TEST")
			self.assertEqual(price, 10.2)
			self.assertEqual(low, 10.1)

	def test_no_data(self):
		# Both intraday and daily empty
		empty = pd.DataFrame()
		with patch("trading_script.yf.download") as mock_dl:
			mock_dl.side_effect = [empty, empty]
			price, low = fetch_intraday_or_last_close("TEST")
			self.assertIsNone(price)
			self.assertIsNone(low)


if __name__ == "__main__":
	unittest.main(verbosity=2)