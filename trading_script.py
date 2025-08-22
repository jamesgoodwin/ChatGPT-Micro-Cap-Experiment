"""Utilities for maintaining the ChatGPT micro cap portfolio.

The script processes portfolio positions, logs trades, and prints daily
results. It is intentionally lightweight and avoids changing existing
logic or behaviour.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any, cast, Optional, Tuple
import os
import time
import warnings

# Suppress FutureWarnings from yfinance
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

# Shared file locations
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # Save files in the same folder as this script
PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"


# === Robust market data helpers ===

def fetch_intraday_or_last_close(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Return a tuple of (current_price, day_low) with reliable fallbacks.

    Strategy:
    1) Try intraday 1m data for today and use the latest close and min low.
    2) Fallback to last 2 daily candles; use last close and low.
    Returns (None, None) if no data is available.
    """
    try:
        intraday = yf.download(ticker, period="1d", interval="1m", auto_adjust=True, progress=False)
        intraday = cast(pd.DataFrame, intraday)
        if not intraday.empty:
            current_price = float(intraday["Close"].iloc[-1].item())
            day_low = float(intraday["Low"].min().item()) if "Low" in intraday.columns else current_price
            return current_price, day_low
    except Exception:
        pass

    try:
        daily = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
        daily = cast(pd.DataFrame, daily)
        if not daily.empty:
            current_price = float(daily["Close"].iloc[-1].item())
            day_low = float(daily["Low"].iloc[-1].item()) if "Low" in daily.columns else current_price
            return current_price, day_low
    except Exception:
        pass

    return None, None


def set_data_dir(data_dir: Path) -> None:
    """Update global paths for portfolio and trade logs.

    Parameters
    ----------
    data_dir:
        Directory where ``chatgpt_portfolio_update.csv`` and
        ``chatgpt_trade_log.csv`` are stored.
    """

    global DATA_DIR, PORTFOLIO_CSV, TRADE_LOG_CSV
    DATA_DIR = Path(data_dir)
    os.makedirs(DATA_DIR, exist_ok=True)
    PORTFOLIO_CSV = DATA_DIR / "chatgpt_portfolio_update.csv"
    TRADE_LOG_CSV = DATA_DIR / "chatgpt_trade_log.csv"

# Today's date reused across logs
today = datetime.today().strftime("%Y-%m-%d")
now = datetime.now()
day = now.weekday()



def process_portfolio(
    portfolio: pd.DataFrame | dict[str, list[object]] | list[dict[str, object]],
    cash: float,
    interactive: bool = True,
) -> tuple[pd.DataFrame, float]:
    """Update daily price information, log stop-loss sells, and prompt for trades.

    Parameters
    ----------
    portfolio:
        Current holdings provided as a DataFrame, mapping of column names to
        lists, or a list of row dictionaries. The input is normalised to a
        ``DataFrame`` before any processing so that downstream code only deals
        with a single type.
    cash:
        Cash balance available for trading.
    interactive:
        When ``True`` (default) the function prompts for manual trades via
        ``input``. Set to ``False`` to skip all interactive prompts – useful
        when the function is driven by a user interface or automated tests.

    Returns
    -------
    tuple[pd.DataFrame, float]
        Updated portfolio and cash balance.
    """
    print(portfolio)
    if isinstance(portfolio, pd.DataFrame):
        portfolio_df = portfolio.copy()
    elif isinstance(portfolio, (dict, list)):
        portfolio_df = pd.DataFrame(portfolio)
    else:  # pragma: no cover - defensive type check
        raise TypeError("portfolio must be a DataFrame, dict, or list of dicts")

    results: list[dict[str, object]] = []
    total_value = 0.0
    total_pnl = 0.0

    if day == 6 or day == 5 and interactive:
        check = input(
            """Today is currently a weekend, so markets were never open.
This will cause the program to calculate data from the last day (usually Friday), and save it as today.
Are you sure you want to do this? To exit, enter 1. """
        )
        if check == "1":
            raise SystemError("Exitting program...")

    if interactive:
        while True:
            action = input(
                f""" You have {cash} in cash.
Would you like to log a manual trade? Enter 'b' for buy, 's' for sell, or press Enter to continue: """
            ).strip().lower()
            if action == "b":
                try:
                    ticker = input("Enter ticker symbol: ").strip().upper()
                    shares = float(input("Enter number of shares: "))
                    buy_price = float(input("Enter buy price: "))
                    stop_loss = float(input("Enter stop loss: "))
                    if shares <= 0 or buy_price <= 0 or stop_loss <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Manual buy cancelled.")
                else:
                    cash, portfolio_df = log_manual_buy(
                        buy_price,
                        shares,
                        ticker,
                        stop_loss,
                        cash,
                        portfolio_df,
                    )
                continue
            if action == "s":
                try:
                    ticker = input("Enter ticker symbol: ").strip().upper()
                    shares = float(input("Enter number of shares to sell: "))
                    sell_price = float(input("Enter sell price: "))
                    if shares <= 0 or sell_price <= 0:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Manual sell cancelled.")
                else:
                    cash, portfolio_df = log_manual_sell(
                        sell_price,
                        shares,
                        ticker,
                        cash,
                        portfolio_df,
                    )
                continue
            break
    print(portfolio_df)
    for _, stock in portfolio_df.iterrows():
        ticker = stock["ticker"]
        shares = int(stock["shares"])
        cost = float(stock["buy_price"])
        cost_basis = float(stock["cost_basis"])
        stop = float(stock["stop_loss"])
        current_price_val, day_low_val = fetch_intraday_or_last_close(ticker)
        if current_price_val is None or day_low_val is None:
            print(f"No data for {ticker}")
            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": "",
                "Total Value": "",
                "PnL": "",
                "Action": "NO DATA",
                "Cash Balance": "",
                "Total Equity": "",
            }
        else:
            low_price = round(float(day_low_val), 2)
            close_price = round(float(current_price_val), 2)

            if low_price <= stop:
                price = stop
                value = round(price * shares, 2)
                pnl = round((price - cost) * shares, 2)
                action = "SELL - Stop Loss Triggered"
                cash += value
                portfolio_df = log_sell(ticker, shares, price, cost, pnl, portfolio_df)
            else:
                price = close_price
                value = round(price * shares, 2)
                pnl = round((price - cost) * shares, 2)
                action = "HOLD"
                total_value += value
                total_pnl += pnl

            row = {
                "Date": today,
                "Ticker": ticker,
                "Shares": shares,
                "Buy Price": cost,
                "Cost Basis": cost_basis,
                "Stop Loss": stop,
                "Current Price": price,
                "Total Value": value,
                "PnL": pnl,
                "Action": action,
                "Cash Balance": "",
                "Total Equity": "",
            }

        results.append(row)

    # Append TOTAL summary row
    total_row = {
        "Date": today,
        "Ticker": "TOTAL",
        "Shares": "",
        "Buy Price": "",
        "Cost Basis": "",
        "Stop Loss": "",
        "Current Price": "",
        "Total Value": round(total_value, 2),
        "PnL": round(total_pnl, 2),
        "Action": "",
        "Cash Balance": round(cash, 2),
        "Total Equity": round(total_value + cash, 2),
    }
    results.append(total_row)

    df = pd.DataFrame(results)
    if PORTFOLIO_CSV.exists():
        existing = pd.read_csv(PORTFOLIO_CSV)
        existing = existing[existing["Date"] != today]
        print("Saving results to CSV...")
        time.sleep(1)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(PORTFOLIO_CSV, index=False)
    return portfolio_df, cash


def log_sell(
    ticker: str,
    shares: float,
    price: float,
    cost: float,
    pnl: float,
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """Record a stop-loss sale in ``TRADE_LOG_CSV`` and remove the ticker."""
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Sold": shares,
        "Sell Price": price,
        "Cost Basis": cost,
        "PnL": pnl,
        "Reason": "AUTOMATED SELL - STOPLOSS TRIGGERED",
    }
    print(f"{ticker} stop loss was met. Selling all shares.")
    portfolio = portfolio[portfolio["ticker"] != ticker]

    if TRADE_LOG_CSV.exists():
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)
    return portfolio


def log_manual_buy(
    buy_price: float,
    shares: float,
    ticker: str,
    stoploss: float,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Log a manual purchase and append to the portfolio."""

    if interactive:
        check = input(
            f"""You are currently trying to buy {shares} shares of {ticker} with a price of {buy_price} and a stoploss of {stoploss}.
        If this a mistake, type "1". """
        )
        if check == "1":
            print("Returning...")
            return cash, chatgpt_portfolio

    # Ensure DataFrame exists with required columns
    if not isinstance(chatgpt_portfolio, pd.DataFrame) or chatgpt_portfolio.empty:
        chatgpt_portfolio = pd.DataFrame(columns=["ticker", "shares", "stop_loss", "buy_price", "cost_basis"])

    # Download current market data
    data = yf.download(ticker, period="1d", auto_adjust=False, progress=False)
    data = cast(pd.DataFrame, data)

    if data.empty:
        print(f"Manual buy for {ticker} failed: no market data available.")
        return cash, chatgpt_portfolio

    day_high = float(data["High"].iloc[-1].item())
    day_low = float(data["Low"].iloc[-1].item())

    if not (day_low <= buy_price <= day_high):
        print(
            f"Manual buy for {ticker} at {buy_price} failed: price outside today's range {round(day_low, 2)}-{round(day_high, 2)}."
        )
        return cash, chatgpt_portfolio

    if buy_price * shares > cash:
        print(
            f"Manual buy for {ticker} failed: cost {buy_price * shares} exceeds cash balance {cash}."
        )
        return cash, chatgpt_portfolio

    # Log trade to trade log CSV
    pnl = 0.0
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": shares,
        "Buy Price": buy_price,
        "Cost Basis": buy_price * shares,
        "PnL": pnl,
        "Reason": "MANUAL BUY - New position",
    }

    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    # === Update portfolio DataFrame ===
    rows = chatgpt_portfolio.loc[
        chatgpt_portfolio["ticker"].astype(str).str.upper() == ticker.upper()
    ]

    if rows.empty:
        # New position
        new_trade = {
            "ticker": ticker,
            "shares": float(shares),
            "stop_loss": float(stoploss),
            "buy_price": float(buy_price),
            "cost_basis": float(buy_price * shares),
        }
        chatgpt_portfolio = pd.concat(
            [chatgpt_portfolio, pd.DataFrame([new_trade])], ignore_index=True
        )
    else:
        # Add to existing position — recompute weighted avg price
        idx = rows.index[0]
        cur_shares = float(chatgpt_portfolio.at[idx, "shares"])
        cur_cost = float(chatgpt_portfolio.at[idx, "cost_basis"])

        new_shares = cur_shares + float(shares)
        new_cost = cur_cost + float(buy_price * shares)
        avg_price = new_cost / new_shares if new_shares else 0.0

        chatgpt_portfolio.at[idx, "shares"] = new_shares
        chatgpt_portfolio.at[idx, "cost_basis"] = new_cost
        chatgpt_portfolio.at[idx, "buy_price"] = avg_price
        chatgpt_portfolio.at[idx, "stop_loss"] = float(stoploss)

    # Deduct cash
    cash -= shares * buy_price
    print(f"Manual buy for {ticker} complete!")
    return cash, chatgpt_portfolio



def log_manual_sell(
    sell_price: float,
    shares_sold: float,
    ticker: str,
    cash: float,
    chatgpt_portfolio: pd.DataFrame,
    reason: str | None = None,
    interactive: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Log a manual sale and update the portfolio.

    Parameters
    ----------
    reason:
        Description of why the position is being sold. Ignored when
        ``interactive`` is ``True``.
    interactive:
        When ``False`` no interactive confirmation is requested.
    """
    if interactive:
        reason = input(
            f"""You are currently trying to sell {shares_sold} shares of {ticker} at a price of {sell_price}.
If this is a mistake, enter 1. """
        )

        if reason == "1":
            print("Returning...")
            return cash, chatgpt_portfolio
    elif reason is None:
        reason = ""
    if ticker not in chatgpt_portfolio["ticker"].values:
        print(f"Manual sell for {ticker} failed: ticker not in portfolio.")
        return cash, chatgpt_portfolio
    ticker_row = chatgpt_portfolio[chatgpt_portfolio["ticker"] == ticker]

    total_shares = int(ticker_row["shares"].item())
    if shares_sold > total_shares:
        print(
            f"Manual sell for {ticker} failed: trying to sell {shares_sold} shares but only own {total_shares}."
        )
        return cash, chatgpt_portfolio
    data = yf.download(ticker, period="1d", auto_adjust=True, progress=False)
    data = cast(pd.DataFrame, data)
    if data.empty:
        print(f"Manual sell for {ticker} failed: no market data available.")
        return cash, chatgpt_portfolio
    day_high = float(data["High"].iloc[-1])
    day_low = float(data["Low"].iloc[-1])
    if not (day_low <= sell_price <= day_high):
        print(
            f"Manual sell for {ticker} at {sell_price} failed: price outside today's range {round(day_low, 2)}-{round(day_high, 2)}."
        )
        return cash, chatgpt_portfolio
    buy_price = float(ticker_row["buy_price"].item())
    cost_basis = buy_price * shares_sold
    pnl = sell_price * shares_sold - cost_basis
    log = {
        "Date": today,
        "Ticker": ticker,
        "Shares Bought": "",
        "Buy Price": "",
        "Cost Basis": cost_basis,
        "PnL": pnl,
        "Reason": f"MANUAL SELL - {reason}",
        "Shares Sold": shares_sold,
        "Sell Price": sell_price,
    }
    if os.path.exists(TRADE_LOG_CSV):
        df = pd.read_csv(TRADE_LOG_CSV)
        df = pd.concat([df, pd.DataFrame([log])], ignore_index=True)
    else:
        df = pd.DataFrame([log])
    df.to_csv(TRADE_LOG_CSV, index=False)

    if total_shares == shares_sold:
        chatgpt_portfolio = chatgpt_portfolio[chatgpt_portfolio["ticker"] != ticker]
    else:
        row_index = ticker_row.index[0]
        chatgpt_portfolio.at[row_index, "shares"] = total_shares - shares_sold
        chatgpt_portfolio.at[row_index, "cost_basis"] = (
            chatgpt_portfolio.at[row_index, "shares"]
            * chatgpt_portfolio.at[row_index, "buy_price"]
        )

    cash = cash + shares_sold * sell_price
    print(f"manual sell for {ticker} complete!")
    return cash, chatgpt_portfolio


def generate_chatgpt_prompt(chatgpt_portfolio: pd.DataFrame, cash: float) -> str:
    """Generate complete ChatGPT prompt including base prompt and current portfolio state."""
    
    # Read the base prompt from Prompts.md
    prompts_file = SCRIPT_DIR / "Experiment Details" / "Prompts.md"
    try:
        with open(prompts_file, 'r', encoding='utf-8') as f:
            base_prompt = f.read().strip()
    except FileNotFoundError:
        base_prompt = "You are a professional-grade portfolio strategist managing a micro-cap stock portfolio."
    
    # Get current portfolio summary
    portfolio_summary = generate_portfolio_summary(chatgpt_portfolio, cash)
    
    # Get market data for key indices
    market_data = generate_market_data()
    
    # Combine everything into the complete prompt
    complete_prompt = f"""{base_prompt}

=== CURRENT PORTFOLIO STATUS ({today}) ===

{portfolio_summary}

=== MARKET DATA ===

{market_data}

=== YOUR TASK ===

Based on the above portfolio status and market conditions, please review my current positions and provide your recommendations for today. You may:
- Hold current positions
- Buy new positions (with available cash)
- Sell existing positions
- Adjust stop-loss levels

Please provide specific trade instructions if you recommend any changes."""

    return complete_prompt


def generate_portfolio_summary(chatgpt_portfolio: pd.DataFrame, cash: float) -> str:
    """Generate a formatted summary of the current portfolio."""
    if chatgpt_portfolio.empty:
        return f"Portfolio: Empty\nCash Balance: ${cash:.2f}\nTotal Equity: ${cash:.2f}"
    
    summary_lines = []
    total_value = 0.0
    total_pnl = 0.0
    
    summary_lines.append("Current Holdings:")
    for _, stock in chatgpt_portfolio.iterrows():
        ticker = stock["ticker"]
        shares = int(stock["shares"])
        buy_price = float(stock["buy_price"])
        cost_basis = float(stock["cost_basis"])
        stop_loss = float(stock["stop_loss"])
        
        # Get current price
        try:
            current_price_val, _ = fetch_intraday_or_last_close(ticker)
            if current_price_val is not None:
                current_price = round(float(current_price_val), 2)
                current_value = round(current_price * shares, 2)
                pnl = round((current_price - buy_price) * shares, 2)
                pnl_pct = round((current_price - buy_price) / buy_price * 100, 2)
                
                total_value += current_value
                total_pnl += pnl
                
                summary_lines.append(f"- {ticker}: {shares} shares @ ${buy_price:.2f} (Current: ${current_price:.2f}) | Value: ${current_value:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.1f}%) | Stop: ${stop_loss:.2f}")
            else:
                summary_lines.append(f"- {ticker}: {shares} shares @ ${buy_price:.2f} | Stop: ${stop_loss:.2f} (No current price data)")
        except Exception:
            summary_lines.append(f"- {ticker}: {shares} shares @ ${buy_price:.2f} | Stop: ${stop_loss:.2f} (Price data unavailable)")
    
    summary_lines.append(f"\nCash Balance: ${cash:.2f}")
    summary_lines.append(f"Total Portfolio Value: ${total_value + cash:.2f}")
    summary_lines.append(f"Total P&L: ${total_pnl:.2f}")
    
    return "\n".join(summary_lines)


def generate_market_data() -> str:
    """Generate formatted market data for key indices."""
    market_lines = []
    
    indices = [
        ("^RUT", "Russell 2000 (Small Caps)"),
        ("IWO", "iShares Russell 2000 Growth ETF"),
        ("XBI", "SPDR S&P Biotech ETF"),
        ("^SPX", "S&P 500")
    ]
    
    for ticker, name in indices:
        try:
            data = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
            data = cast(pd.DataFrame, data)
            if data.empty or len(data) < 2:
                market_lines.append(f"{name} ({ticker}): Data unavailable")
                continue
                
            current_price = float(data["Close"].iloc[-1].item())
            prev_price = float(data["Close"].iloc[-2].item())
            percent_change = ((current_price - prev_price) / prev_price) * 100
            volume = float(data["Volume"].iloc[-1].item()) if "Volume" in data.columns else 0
            
            market_lines.append(f"{name} ({ticker}): ${current_price:.2f} ({percent_change:+.2f}%) | Volume: {volume:,.0f}")
            
        except Exception as e:
            market_lines.append(f"{name} ({ticker}): Error retrieving data")
    
    return "\n".join(market_lines)


def daily_results(chatgpt_portfolio: pd.DataFrame, cash: float) -> None:
    """Print daily price updates and performance metrics."""
    portfolio_dict: list[dict[str, object]] = chatgpt_portfolio.to_dict(orient="records")

    print(f"prices and updates for {today}")
    for stock in portfolio_dict + [{"ticker": "^RUT"}] + [{"ticker": "IWO"}] + [{"ticker": "XBI"}]:
        ticker = stock["ticker"]
        try:
            data = yf.download(ticker, period="2d", auto_adjust=True, progress=False)
            data = cast(pd.DataFrame, data)
            if data.empty or len(data) < 2:
                print(f"Data for {ticker} was empty or incomplete.")
                continue
            price = float(data["Close"].iloc[-1].item())
            last_price = float(data["Close"].iloc[-2].item())

            percent_change = ((price - last_price) / last_price) * 100
            volume = float(data["Volume"].iloc[-1].item())
        except Exception as e:
            raise Exception(f"Download for {ticker} failed. {e} Try checking internet connection.")
        print(f"{ticker} closing price: {price:.2f}")
        print(f"{ticker} volume for today: ${volume:,}")
        print(f"percent change from the day before: {percent_change:.2f}%")
    chatgpt_df = pd.read_csv(PORTFOLIO_CSV)

# Use only TOTAL rows, sorted by date
    totals = chatgpt_df[chatgpt_df["Ticker"] == "TOTAL"].copy()
    totals["Date"] = pd.to_datetime(totals["Date"])
    totals = totals.sort_values("Date")
    final_equity = float(totals.iloc[-1]["Total Equity"])
    equity = totals["Total Equity"].astype(float).reset_index(drop=True)

# Daily simple returns
    r = equity.pct_change().dropna()
    n_days = len(r)

# Config
    rf_annual = 0.045

# Risk-free aligned to frequency and window
    rf_daily  = (1 + rf_annual)**(1 / 252) - 1
    rf_period = (1 + rf_daily)**n_days - 1

# Stats
    mean_daily = r.mean()
    std_daily  = r.std(ddof=1)

# Downside deviation vs MAR = rf_daily
    downside = (r - rf_daily).clip(upper=0)
    downside_std = (downside.pow(2).mean())**0.5

# total return over the window
    period_return = (1 + r).prod() - 1

# --- Sharpe ---
    sharpe_period = (period_return - rf_period) / (std_daily * np.sqrt(n_days))
    sharpe_annual = ((mean_daily - rf_daily) / std_daily) * np.sqrt(252)

# --- Sortino ---
    sortino_period = (period_return - rf_period) / (downside_std * np.sqrt(n_days))
    sortino_annual = ((mean_daily - rf_daily) / downside_std) * np.sqrt(252)

    # Output
    print(f"Total Sharpe Ratio over {n_days} days: {sharpe_period:.4f}")
    print(f"Total Sortino Ratio over {n_days} days: {sortino_period:.4f}")
    print(f"Annualized Sharpe Ratio: {sharpe_annual:.4f}")
    print(f"Annualized Sortino Ratio: {sortino_annual:.4f}")
    print(f"Latest ChatGPT Equity: ${final_equity:.2f}")
    # Get S&P 500 data
    final_date = totals.loc[totals.index[-1], "Date"]
    spx = yf.download("^SPX", start="2025-06-27", end=final_date + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    spx = cast(pd.DataFrame, spx)
    spx = spx.reset_index()

    # Normalize to $100
    initial_price = spx["Close"].iloc[0].item()
    price_now = spx["Close"].iloc[-1].item()
    scaling_factor = 100 / initial_price
    spx_value = price_now * scaling_factor
    print(f"$100 Invested in the S&P 500: ${spx_value:.2f}")
    print("today's portfolio:")
    print(chatgpt_portfolio)
    print(f"cash balance: {cash}")

    print("\n" + "="*80)
    print("COMPLETE CHATGPT PROMPT - COPY AND PASTE BELOW:")
    print("="*80)
    
    # Generate and display the complete prompt
    complete_prompt = generate_chatgpt_prompt(chatgpt_portfolio, cash)
    print(complete_prompt)
    
    print("\n" + "="*80)
    print("END OF PROMPT")
    print("="*80)


def main(file: str, data_dir: Path | None = None) -> None:
    """Run the trading script.

    Parameters
    ----------
    file:
        CSV file containing historical portfolio records.
    data_dir:
        Directory where trade and portfolio CSVs will be stored.
    """
    chatgpt_portfolio, cash = load_latest_portfolio_state(file)
    if data_dir is not None:
        set_data_dir(data_dir)

    # Run in non-interactive mode to support automated/scripted execution
    chatgpt_portfolio, cash = process_portfolio(chatgpt_portfolio, cash, interactive=False)
    daily_results(chatgpt_portfolio, cash)

def load_latest_portfolio_state(
    file: str,
) -> tuple[pd.DataFrame | list[dict[str, Any]], float]:
    """Load the most recent portfolio snapshot and cash balance.

    Parameters
    ----------
    file:
        CSV file containing historical portfolio records.

    Returns
    -------
    tuple[pd.DataFrame | list[dict[str, Any]], float]
        A representation of the latest holdings (either an empty DataFrame or a
        list of row dictionaries) and the associated cash balance.
    """

    df = pd.read_csv(file)
    if df.empty:
        portfolio = pd.DataFrame([])
        print(
            "Portfolio CSV is empty. Returning set amount of cash for creating portfolio."
        )
        try:
            cash = float(input("What would you like your starting cash amount to be? "))
        except ValueError:
            raise ValueError(
                "Cash could not be converted to float datatype. Please enter a valid number."
            )
        return portfolio, cash
    non_total = df[df["Ticker"] != "TOTAL"].copy()
    non_total["Date"] = pd.to_datetime(non_total["Date"])

    latest_date = non_total["Date"].max()
    print(latest_date)
    # Get all tickers from the latest date
    latest_tickers = non_total[non_total["Date"] == latest_date].copy()
    sold_mask = latest_tickers["Action"].astype(str).str.startswith("SELL")
    latest_tickers = latest_tickers[~sold_mask].copy()
    latest_tickers.drop(columns=["Date", "Cash Balance", "Total Equity", "Action", "Current Price", "PnL", "Total Value"], inplace=True)
    latest_tickers.rename(columns={"Cost Basis": "cost_basis", "Buy Price": "buy_price", "Shares": "shares", "Ticker": "ticker", "Stop Loss": "stop_loss"}, inplace=True)
    print(latest_tickers)
    latest_tickers = latest_tickers.reset_index(drop=True).to_dict(orient='records')
    df = df[df["Ticker"] == "TOTAL"]  # Only the total summary rows
    df["Date"] = pd.to_datetime(df["Date"])
    latest = df.sort_values("Date").iloc[-1]
    cash = float(latest["Cash Balance"])
    print(latest_tickers)
    return latest_tickers, cash

