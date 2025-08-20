#!/usr/bin/env python3
"""
Test script to verify Interactive Brokers API connection
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_ib_connection():
    """Test basic connection to Interactive Brokers API"""
    
    # Get credentials
    username = os.getenv('IB_USERNAME')
    password = os.getenv('IB_PASSWORD')
    account_id = os.getenv('IB_ACCOUNT_ID')
    
    if not all([username, password, account_id]):
        print("❌ Missing IB credentials in environment variables")
        print("Please set IB_USERNAME, IB_PASSWORD, and IB_ACCOUNT_ID")
        return False
    
    print("🔍 Testing Interactive Brokers API connection...")
    print(f"Username: {username}")
    print(f"Account ID: {account_id}")
    
    try:
        # Test basic connectivity
        base_url = "https://www.interactivebrokers.com/portal.proxy/v1/portal"
        
        # Test authentication endpoint
        auth_url = f"{base_url}/iserver/auth/ssodh/init"
        auth_data = {
            "username": username,
            "password": password
        }
        
        print("📡 Attempting authentication...")
        response = requests.post(auth_url, json=auth_data, timeout=30)
        
        if response.status_code == 200:
            auth_result = response.json()
            if auth_result.get('authenticated'):
                print("✅ Authentication successful!")
                
                # Test account access
                session = requests.Session()
                session.headers.update(response.headers)
                
                accounts_url = f"{base_url}/iserver/accounts"
                accounts_response = session.get(accounts_url)
                
                if accounts_response.status_code == 200:
                    accounts = accounts_response.json()
                    print(f"✅ Found {len(accounts)} accounts")
                    
                    # Check if our account is accessible
                    for account in accounts:
                        if account.get('accountId') == account_id:
                            print(f"✅ Account {account_id} is accessible")
                            return True
                    
                    print(f"❌ Account {account_id} not found in accessible accounts")
                    return False
                else:
                    print(f"❌ Failed to access accounts: {accounts_response.status_code}")
                    return False
            else:
                print(f"❌ Authentication failed: {auth_result}")
                return False
        else:
            print(f"❌ Authentication request failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_ib_connection()
    if success:
        print("🎉 IB API connection test passed!")
        print("You can now run the full portfolio checker script.")
    else:
        print("❌ IB API connection test failed!")
        print("Please check your credentials and try again.")
