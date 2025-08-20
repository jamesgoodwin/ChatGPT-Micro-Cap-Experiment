#!/bin/bash
# Pre-push test script for GitHub Actions workflows
# Run this before pushing to catch issues early

set -e  # Exit on any error

echo "🚀 Running pre-push tests..."
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Run unit tests
echo "🧪 Running unit tests..."
python test_portfolio_health.py

# Run comprehensive workflow tests
echo "🔧 Running workflow tests..."
python test_workflow_locally.py

# Check git status
echo "📋 Checking git status..."
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  You have uncommitted changes:"
    git status --short
    echo ""
    echo "💡 Consider committing your changes before pushing:"
    echo "   git add ."
    echo "   git commit -m 'Your commit message'"
    echo "   git push"
else
    echo "✅ No uncommitted changes"
fi

echo ""
echo "🎉 Pre-push tests completed successfully!"
echo "   You can now safely push to GitHub."
