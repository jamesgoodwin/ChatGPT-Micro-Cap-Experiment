#!/usr/bin/env python3
"""
Interactive Brokers Portfolio Checker
Connects to IB Web API to fetch real account portfolio data
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ib_portfolio.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IBPortfolioChecker:
    """Interactive Brokers Portfolio Checker using Web API"""
    
    def __init__(self):
        """Initialize the IB Portfolio Checker"""
        self.base_url = "https://www.interactivebrokers.com/portal.proxy/v1/portal"
        self.session = requests.Session()
        self.session_id = None
        self.conid = None
        
        # Load credentials from environment variables
        self.username = os.getenv('IB_USERNAME')
        self.password = os.getenv('IB_PASSWORD')
        self.account_id = os.getenv('IB_ACCOUNT_ID')
        
        if not all([self.username, self.password, self.account_id]):
            logger.error("Missing IB credentials in environment variables")
            logger.info("Please set IB_USERNAME, IB_PASSWORD, and IB_ACCOUNT_ID")
            raise ValueError("Missing IB credentials")
    
    def authenticate(self) -> bool:
        """Authenticate with Interactive Brokers Web API"""
        try:
            logger.info("Authenticating with Interactive Brokers...")
            
            # Step 1: Get session ID
            auth_url = f"{self.base_url}/iserver/auth/ssodh/init"
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            response = self.session.post(auth_url, json=auth_data)
            response.raise_for_status()
            
            auth_result = response.json()
            if auth_result.get('authenticated'):
                logger.info("✅ Authentication successful")
                return True
            else:
                logger.error(f"❌ Authentication failed: {auth_result}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        try:
            logger.info("Fetching account information...")
            
            # Get account list
            accounts_url = f"{self.base_url}/iserver/accounts"
            response = self.session.get(accounts_url)
            response.raise_for_status()
            
            accounts = response.json()
            logger.info(f"Found {len(accounts)} accounts")
            
            # Find our specific account
            for account in accounts:
                if account.get('accountId') == self.account_id:
                    logger.info(f"✅ Found account: {account.get('accountId')}")
                    return account
            
            logger.error(f"❌ Account {self.account_id} not found")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching account info: {e}")
            return None
    
    def get_portfolio_positions(self) -> Optional[List[Dict]]:
        """Get current portfolio positions"""
        try:
            logger.info("Fetching portfolio positions...")
            
            positions_url = f"{self.base_url}/iserver/account/{self.account_id}/positions"
            response = self.session.get(positions_url)
            response.raise_for_status()
            
            positions = response.json()
            logger.info(f"Found {len(positions)} positions")
            
            return positions
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching positions: {e}")
            return None
    
    def get_account_summary(self) -> Optional[Dict]:
        """Get account summary including cash, buying power, etc."""
        try:
            logger.info("Fetching account summary...")
            
            summary_url = f"{self.base_url}/iserver/account/{self.account_id}/summary"
            response = self.session.get(summary_url)
            response.raise_for_status()
            
            summary = response.json()
            logger.info("✅ Account summary retrieved")
            
            return summary
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching account summary: {e}")
            return None
    
    def get_market_data(self, conids: List[str]) -> Optional[Dict]:
        """Get market data for specific contract IDs"""
        try:
            logger.info(f"Fetching market data for {len(conids)} symbols...")
            
            # Get snapshot data
            snapshot_url = f"{self.base_url}/iserver/marketdata/snapshot"
            params = {
                "conids": ",".join(conids)
            }
            
            response = self.session.get(snapshot_url, params=params)
            response.raise_for_status()
            
            market_data = response.json()
            logger.info("✅ Market data retrieved")
            
            return market_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching market data: {e}")
            return None
    
    def format_portfolio_data(self, positions: List[Dict], account_summary: Dict) -> pd.DataFrame:
        """Format portfolio data into a pandas DataFrame"""
        try:
            portfolio_data = []
            
            for position in positions:
                # Extract position data
                ticker = position.get('ticker', '')
                conid = position.get('conid', '')
                size = position.get('position', 0)
                market_value = position.get('marketValue', 0)
                unrealized_pnl = position.get('unrealizedPnl', 0)
                cost_basis = position.get('costBasis', 0)
                
                # Calculate additional metrics
                avg_price = cost_basis / size if size != 0 else 0
                current_price = market_value / size if size != 0 else 0
                pnl_percent = (unrealized_pnl / cost_basis * 100) if cost_basis != 0 else 0
                
                portfolio_data.append({
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'ConID': conid,
                    'Shares': size,
                    'Market_Value': market_value,
                    'Cost_Basis': cost_basis,
                    'Avg_Price': avg_price,
                    'Current_Price': current_price,
                    'Unrealized_PnL': unrealized_pnl,
                    'PnL_Percent': pnl_percent,
                    'Action': 'HOLD'  # Default action
                })
            
            # Add cash position
            cash_balance = account_summary.get('cashBalance', 0)
            portfolio_data.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Ticker': 'CASH',
                'ConID': 'CASH',
                'Shares': 1,
                'Market_Value': cash_balance,
                'Cost_Basis': cash_balance,
                'Avg_Price': cash_balance,
                'Current_Price': cash_balance,
                'Unrealized_PnL': 0,
                'PnL_Percent': 0,
                'Action': 'CASH'
            })
            
            # Create DataFrame
            df = pd.DataFrame(portfolio_data)
            
            # Add summary row
            total_market_value = df['Market_Value'].sum()
            total_cost_basis = df['Cost_Basis'].sum()
            total_pnl = df['Unrealized_PnL'].sum()
            total_pnl_percent = (total_pnl / total_cost_basis * 100) if total_cost_basis != 0 else 0
            
            summary_row = {
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Ticker': 'TOTAL',
                'ConID': 'TOTAL',
                'Shares': '',
                'Market_Value': total_market_value,
                'Cost_Basis': total_cost_basis,
                'Avg_Price': '',
                'Current_Price': '',
                'Unrealized_PnL': total_pnl,
                'PnL_Percent': total_pnl_percent,
                'Action': 'SUMMARY'
            }
            
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error formatting portfolio data: {e}")
            return pd.DataFrame()
    
    def save_portfolio_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save portfolio data to CSV file"""
        try:
            if filename is None:
                filename = f"ib_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            filepath = os.path.join(os.path.dirname(__file__), filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ Portfolio data saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error saving portfolio data: {e}")
            return ""
    
    def generate_portfolio_report(self, df: pd.DataFrame, account_summary: Dict) -> str:
        """Generate a comprehensive portfolio report"""
        try:
            report = []
            report.append("# Interactive Brokers Portfolio Report")
            report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
            report.append("")
            
            # Account Summary
            report.append("## Account Summary")
            report.append(f"- **Account ID:** {self.account_id}")
            report.append(f"- **Cash Balance:** ${account_summary.get('cashBalance', 0):,.2f}")
            report.append(f"- **Buying Power:** ${account_summary.get('buyingPower', 0):,.2f}")
            report.append(f"- **Net Liquidation Value:** ${account_summary.get('netLiquidationValue', 0):,.2f}")
            report.append("")
            
            # Portfolio Summary
            positions_df = df[df['Ticker'] != 'TOTAL']
            positions_df = positions_df[positions_df['Ticker'] != 'CASH']
            
            if not positions_df.empty:
                report.append("## Portfolio Positions")
                report.append("")
                
                for _, row in positions_df.iterrows():
                    report.append(f"### {row['Ticker']}")
                    report.append(f"- **Shares:** {row['Shares']:,.0f}")
                    report.append(f"- **Market Value:** ${row['Market_Value']:,.2f}")
                    report.append(f"- **Cost Basis:** ${row['Cost_Basis']:,.2f}")
                    report.append(f"- **Current Price:** ${row['Current_Price']:.2f}")
                    report.append(f"- **Avg Price:** ${row['Avg_Price']:.2f}")
                    report.append(f"- **Unrealized P&L:** ${row['Unrealized_PnL']:,.2f} ({row['PnL_Percent']:+.2f}%)")
                    report.append("")
            
            # Total Summary
            total_row = df[df['Ticker'] == 'TOTAL'].iloc[0]
            report.append("## Portfolio Summary")
            report.append(f"- **Total Market Value:** ${total_row['Market_Value']:,.2f}")
            report.append(f"- **Total Cost Basis:** ${total_row['Cost_Basis']:,.2f}")
            report.append(f"- **Total Unrealized P&L:** ${total_row['Unrealized_PnL']:,.2f} ({total_row['PnL_Percent']:+.2f}%)")
            report.append("")
            
            report.append("---")
            report.append("*Report generated by IB Portfolio Checker*")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return f"Error generating report: {e}"
    
    def run_portfolio_check(self) -> bool:
        """Main method to run the complete portfolio check"""
        try:
            logger.info("🚀 Starting Interactive Brokers Portfolio Check")
            
            # Step 1: Authenticate
            if not self.authenticate():
                return False
            
            # Step 2: Get account info
            account_info = self.get_account_info()
            if not account_info:
                return False
            
            # Step 3: Get portfolio positions
            positions = self.get_portfolio_positions()
            if positions is None:
                return False
            
            # Step 4: Get account summary
            account_summary = self.get_account_summary()
            if not account_summary:
                return False
            
            # Step 5: Format data
            portfolio_df = self.format_portfolio_data(positions, account_summary)
            if portfolio_df.empty:
                logger.error("❌ No portfolio data to process")
                return False
            
            # Step 6: Save data
            csv_file = self.save_portfolio_data(portfolio_df, "ib_portfolio_update.csv")
            if not csv_file:
                return False
            
            # Step 7: Generate report
            report = self.generate_portfolio_report(portfolio_df, account_summary)
            report_file = os.path.join(os.path.dirname(__file__), "ib_portfolio_report.md")
            
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"✅ Report saved to: {report_file}")
            
            # Step 8: Print summary
            total_row = portfolio_df[portfolio_df['Ticker'] == 'TOTAL'].iloc[0]
            logger.info("📊 Portfolio Summary:")
            logger.info(f"   Total Market Value: ${total_row['Market_Value']:,.2f}")
            logger.info(f"   Total P&L: ${total_row['Unrealized_PnL']:,.2f} ({total_row['PnL_Percent']:+.2f}%)")
            logger.info(f"   Cash Balance: ${account_summary.get('cashBalance', 0):,.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in portfolio check: {e}")
            return False

def main():
    """Main function to run the IB Portfolio Checker"""
    try:
        # Create portfolio checker instance
        checker = IBPortfolioChecker()
        
        # Run the portfolio check
        success = checker.run_portfolio_check()
        
        if success:
            logger.info("🎉 Portfolio check completed successfully!")
            return 0
        else:
            logger.error("❌ Portfolio check failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
