# Interactive Brokers Portfolio Checker

This script connects to your Interactive Brokers account via the Web API to fetch real portfolio data, including positions, account balances, and performance metrics.

## Features

- 🔐 **Secure Authentication** - Uses IB Web API with your credentials
- 📊 **Real Portfolio Data** - Fetches actual positions and account balances
- 💰 **Performance Tracking** - Calculates P&L, cost basis, and performance metrics
- 📈 **Market Data** - Gets current prices and market values
- 📄 **Report Generation** - Creates comprehensive portfolio reports
- 💾 **Data Export** - Saves data to CSV format for analysis
- 📝 **Logging** - Comprehensive logging for debugging

## Setup Instructions

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Set Up Credentials

Create a `.env` file in the project root with your Interactive Brokers credentials:

```bash
# Copy the example file
cp env_example.txt .env

# Edit the .env file with your actual credentials
nano .env
```

Fill in your credentials:

```env
# Your Interactive Brokers username
IB_USERNAME=your_ib_username

# Your Interactive Brokers password
IB_PASSWORD=your_ib_password

# Your Interactive Brokers account ID (e.g., U1234567)
IB_ACCOUNT_ID=your_account_id
```

### 3. Test Connection

Before running the full script, test your connection:

```bash
cd "Scripts and CSV Files"
python test_ib_connection.py
```

If successful, you should see:
```
✅ Authentication successful!
✅ Found X accounts
✅ Account your_account_id is accessible
🎉 IB API connection test passed!
```

### 4. Run Portfolio Check

Once the connection test passes, run the full portfolio checker:

```bash
python ib_portfolio_checker.py
```

## Output Files

The script generates several output files:

### 1. Portfolio Data (CSV)
- **File:** `ib_portfolio_update.csv`
- **Contains:** All positions with market values, P&L, cost basis, etc.

### 2. Portfolio Report (Markdown)
- **File:** `ib_portfolio_report.md`
- **Contains:** Formatted report with account summary and position details

### 3. Log File
- **File:** `ib_portfolio.log`
- **Contains:** Detailed execution logs for debugging

## Sample Output

### Console Output
```
🚀 Starting Interactive Brokers Portfolio Check
✅ Authentication successful
✅ Found account: U1234567
Found 5 positions
✅ Account summary retrieved
✅ Portfolio data saved to: ib_portfolio_update.csv
✅ Report saved to: ib_portfolio_report.md

📊 Portfolio Summary:
   Total Market Value: $125,432.67
   Total P&L: $2,345.89 (+1.91%)
   Cash Balance: $15,234.56
```

### CSV Data Structure
| Date | Ticker | ConID | Shares | Market_Value | Cost_Basis | Avg_Price | Current_Price | Unrealized_PnL | PnL_Percent | Action |
|------|--------|-------|--------|--------------|------------|-----------|---------------|----------------|-------------|--------|
| 2025-08-20 | AAPL | 76792991 | 100 | 17500.00 | 17000.00 | 170.00 | 175.00 | 500.00 | 2.94 | HOLD |
| 2025-08-20 | CASH | CASH | 1 | 15234.56 | 15234.56 | 15234.56 | 15234.56 | 0.00 | 0.00 | CASH |
| 2025-08-20 | TOTAL | TOTAL | | 140734.56 | 138234.56 | | | 2500.00 | 1.81 | SUMMARY |

## API Endpoints Used

The script uses the following Interactive Brokers Web API endpoints:

- **Authentication:** `/iserver/auth/ssodh/init`
- **Accounts:** `/iserver/accounts`
- **Positions:** `/iserver/account/{accountId}/positions`
- **Account Summary:** `/iserver/account/{accountId}/summary`
- **Market Data:** `/iserver/marketdata/snapshot`

## Security Notes

- **Credentials:** Never commit your `.env` file to version control
- **API Access:** The script uses read-only API endpoints
- **Session Management:** Sessions are handled securely with proper headers
- **Logging:** Sensitive data is not logged

## Troubleshooting

### Common Issues

1. **Authentication Failed**
   - Verify your username and password
   - Check if your account has API access enabled
   - Ensure you're not using 2FA (Web API doesn't support it)

2. **Account Not Found**
   - Verify your account ID format (e.g., U1234567)
   - Check if the account is accessible with your credentials

3. **Connection Timeout**
   - Check your internet connection
   - IB servers might be temporarily unavailable

4. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Ensure you're using Python 3.7+

### Debug Mode

For detailed debugging, check the log file:

```bash
tail -f ib_portfolio.log
```

## Integration with Existing Workflows

The IB Portfolio Checker can be integrated with your existing trading workflows:

1. **Replace Simulated Data:** Use real IB data instead of simulated portfolio data
2. **Automated Updates:** Schedule regular portfolio checks
3. **Performance Tracking:** Compare simulated vs. actual performance
4. **Risk Management:** Monitor real account exposure

## License

This script is part of the ChatGPT Micro Cap Experiment project and follows the same licensing terms.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the log files for error details
3. Verify your IB account has API access enabled
4. Contact Interactive Brokers support for API-related issues
