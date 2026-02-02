#!/bin/bash
# setup_geo_features.sh
# Quick setup script to enable geographical APIs and structured datasets

echo "=== Automated Research System - Geo & Datasets Setup ==="
echo

# Check if .env exists
if [ -f ".env" ]; then
    echo "✓ .env file found"
else
    echo "⚠ Creating .env file..."
    touch .env
fi

echo
echo "Adding geographical API configuration..."

# Function to add or update env variable
add_or_update_env() {
    local key=$1
    local value=$2
    
    if grep -q "^${key}=" .env; then
        # Update existing
        sed -i "s/^${key}=.*/${key}=${value}/" .env
        echo "  ✓ Updated $key=$value"
    else
        # Add new
        echo "${key}=${value}" >> .env
        echo "  ✓ Added $key=$value"
    fi
}

echo
echo "Configuration Options:"
echo "====================="
echo
echo "1. ENABLE GEO VERIFICATION"
echo "   Makes location verification automatic during extraction"
read -p "   Enable geo verification? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    add_or_update_env "USE_GEO_VERIFICATION" "true"
else
    add_or_update_env "USE_GEO_VERIFICATION" "false"
fi

echo
echo "2. CREATE STRUCTURED DATASETS"
echo "   Automatically export results as structured datasets (JSON, CSV, etc)"
read -p "   Enable structured datasets? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    add_or_update_env "CREATE_STRUCTURED_DATASETS" "true"
else
    add_or_update_env "CREATE_STRUCTURED_DATASETS" "false"
fi

echo
echo "3. GEO VERIFICATION CONFIDENCE THRESHOLD"
echo "   Minimum confidence score for geo-verified locations (0.0-1.0)"
read -p "   Enter minimum confidence (default 0.7): " confidence
confidence=${confidence:-0.7}
add_or_update_env "GEO_MIN_CONFIDENCE" "$confidence"

echo
echo "4. OPTIONAL: LocationIQ API Key (for higher rate limits)"
echo "   Get free tier at: https://locationiq.com/sign-up"
read -p "   Enter LocationIQ API key (or press Enter to skip): " locationiq_key
if [ -n "$locationiq_key" ]; then
    add_or_update_env "LOCATIONIQ_API_KEY" "$locationiq_key"
    echo "  ✓ Added LocationIQ API key"
else
    echo "  ℹ Using free Nominatim (1 req/sec)"
fi

echo
echo "=== Python Dependencies ==="
echo
echo "Required packages:"
echo "  - requests (for HTTP API calls)"
echo "  - sentence-transformers (already installed)"
echo "  - faiss-cpu (for vector search, already installed)"
echo
echo "Optional packages (for enhanced functionality):"
echo "  - pandas (for dataset analysis)"
echo "  - pyarrow (for Parquet export)"
echo
read -p "Install optional dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install pandas pyarrow
    echo "✓ Optional dependencies installed"
fi

echo
echo "=== Configuration Complete ==="
echo
echo "Current .env settings:"
grep -E "(USE_GEO_VERIFICATION|CREATE_STRUCTURED_DATASETS|GEO_MIN_CONFIDENCE|LOCATIONIQ)" .env || echo "  (none set)"

echo
echo "Next steps:"
echo "1. Verify .env file looks correct: cat .env"
echo "2. Run example: python -c 'from geo_verifier import create_verifier; v = create_verifier(); print(\"✓ Geo module loaded\")'"
echo "3. Try demo notebook: jupyter notebook geo_datasets_demo.ipynb"
echo "4. Run research task with: python main.py"
echo

# Optional: Run a quick test
read -p "Run quick validation test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing imports..."
    python3 << 'EOF'
try:
    from geo_verifier import create_verifier
    from dataset_builder import create_hospital_dataset
    print("✓ All modules imported successfully")
    
    # Quick test
    verifier = create_verifier()
    print("✓ GeoVerifier initialized")
    
    builder = create_hospital_dataset()
    print("✓ Dataset builder initialized")
    
    print("\n✅ All tests passed! Ready to use geo features.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check your Python environment and dependencies")
EOF
fi

echo
echo "Documentation: See GEO_DATASETS_GUIDE.md for full usage guide"
