#!/usr/bin/env python3
"""Local testing script for GitHub Actions workflow steps."""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"   ✅ Success")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed with exit code {e.returncode}")
        if e.stdout.strip():
            print(f"   Stdout: {e.stdout.strip()}")
        if e.stderr.strip():
            print(f"   Stderr: {e.stderr.strip()}")
        return False

def test_python_imports():
    """Test that all required Python modules can be imported."""
    print("\n📦 Testing Python imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        import matplotlib.pyplot as plt
        import requests
        from trading_script import (
            generate_chatgpt_prompt, 
            generate_portfolio_summary, 
            generate_market_data,
            load_latest_portfolio_state
        )
        print("   ✅ All imports successful")
        return True
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_portfolio_health_check():
    """Test the portfolio health check functionality."""
    print("\n🏥 Testing portfolio health check...")
    
    # Create test CSV file
    test_data = [
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        import pandas as pd
        df = pd.DataFrame(test_data)
        df.to_csv(f.name, index=False)
        test_csv_path = f.name
    
    try:
        # Test the portfolio health check function
        import pandas as pd
        from datetime import datetime
        
        def check_portfolio_health(csv_path):
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
        
        result = check_portfolio_health(test_csv_path)
        if result:
            print("   ✅ Portfolio health check passed")
            return True
        else:
            print("   ❌ Portfolio health check failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Portfolio health check error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_csv_path):
            os.unlink(test_csv_path)

def test_trading_script_functions():
    """Test the trading script functions."""
    print("\n📈 Testing trading script functions...")
    
    try:
        from trading_script import generate_chatgpt_prompt, generate_portfolio_summary, generate_market_data
        import pandas as pd
        
        # Test with empty portfolio
        empty_portfolio = pd.DataFrame()
        cash = 100.0
        
        # Test portfolio summary
        summary = generate_portfolio_summary(empty_portfolio, cash)
        print(f"   ✅ Portfolio summary generated ({len(summary)} chars)")
        
        # Test market data
        market_data = generate_market_data()
        print(f"   ✅ Market data generated ({len(market_data)} chars)")
        
        # Test ChatGPT prompt
        prompt = generate_chatgpt_prompt(empty_portfolio, cash)
        print(f"   ✅ ChatGPT prompt generated ({len(prompt)} chars)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Trading script functions error: {e}")
        return False

def test_openai_api_integration():
    """Test OpenAI API integration (if API key is available)."""
    print("\n🤖 Testing OpenAI API integration...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("   ⚠️  OPENAI_API_KEY not set, skipping API test")
        return True
    
    try:
        import requests
        from trading_script import generate_chatgpt_prompt
        import pandas as pd
        
        # Create test portfolio
        portfolio_data = [
            {
                'ticker': 'ABEO',
                'shares': 1,
                'buy_price': 5.87,
                'cost_basis': 5.87,
                'stop_loss': 4.50
            }
        ]
        portfolio = pd.DataFrame(portfolio_data)
        cash = 94.13
        
        # Generate prompt
        prompt = generate_chatgpt_prompt(portfolio, cash)
        
        # Test API call
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-4',
            'messages': [
                {
                    'role': 'user', 
                    'content': prompt[:100] + "..."  # Truncate for testing
                }
            ],
            'max_tokens': 50,
            'temperature': 0.7
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("   ✅ OpenAI API call successful")
            return True
        else:
            print(f"   ❌ OpenAI API call failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ OpenAI API integration error: {e}")
        return False

def test_yaml_syntax():
    """Test YAML syntax of workflow files."""
    print("\n📄 Testing YAML syntax...")
    
    workflow_files = [
        '.github/workflows/portfolio-check.yml',
        '.github/workflows/release.yml',
        '.github/workflows/manual-update.yml'
    ]
    
    all_passed = True
    for workflow_file in workflow_files:
        if os.path.exists(workflow_file):
            try:
                import yaml
                with open(workflow_file, 'r') as f:
                    yaml.safe_load(f)
                print(f"   ✅ {workflow_file} - Valid YAML")
            except Exception as e:
                print(f"   ❌ {workflow_file} - Invalid YAML: {e}")
                all_passed = False
        else:
            print(f"   ⚠️  {workflow_file} - File not found")
    
    return all_passed

def main():
    """Run all local tests."""
    print("🚀 Running local workflow tests...")
    print("=" * 50)
    
    tests = [
        ("Python imports", test_python_imports),
        ("YAML syntax", test_yaml_syntax),
        ("Portfolio health check", test_portfolio_health_check),
        ("Trading script functions", test_trading_script_functions),
        ("OpenAI API integration", test_openai_api_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ {test_name} - Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Safe to push to GitHub.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before pushing.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
