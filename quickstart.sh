#!/bin/bash
# Quick Start Script for Dataset Builder

set -e

echo "=================================="
echo "Dataset Builder - Quick Start"
echo "=================================="
echo ""

# Check Python environment
echo "✓ Checking Python environment..."
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

# Activate conda environment
echo "✓ Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate auto_research 2>/dev/null || {
    echo "⚠️  auto_research environment not found. Creating it..."
    conda create -n auto_research python=3.12 -y
    conda activate auto_research
}

# Install Playwright
echo "✓ Installing Playwright..."
pip install playwright -q || true
python -m playwright install chromium -q || true

# Check environment variables
echo ""
echo "✓ Checking environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='your-key'"
    exit 1
fi

if [ -z "$SEARXNG_URL" ]; then
    echo "⚠️  SEARXNG_URL not set. Using default: http://localhost:8888"
    export SEARXNG_URL="http://localhost:8888"
fi

# Start SEARXNG in background (optional)
echo ""
read -p "Start SEARXNG? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "✓ Starting SEARXNG..."
    cd searxng-docker
    docker-compose up -d
    cd ..
    sleep 5
    echo "✓ SEARXNG started"
fi

# Start server
echo ""
echo "✓ Starting Research Server..."
echo ""
echo "=================================="
echo "🚀 Server running on http://localhost:8000"
echo "=================================="
echo ""

python server.py
