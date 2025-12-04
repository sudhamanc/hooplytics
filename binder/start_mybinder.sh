#!/bin/bash

echo "========================================="
echo "🏀 Hooplytics - MyBinder Setup"
echo "========================================="

# Check for API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: GOOGLE_API_KEY not set!"
    echo ""
    echo "To use Hooplytics, you need a Google Gemini API key."
    echo ""
    echo "Steps to set it up:"
    echo "1. Get your API key from: https://makersuite.google.com/app/apikey"
    echo "2. Run this command in the terminal:"
    echo "   export GOOGLE_API_KEY='your-api-key-here'"
    echo "3. Then restart this script: ./binder/start_mybinder.sh"
    echo ""
    echo "========================================="
    read -p "Press Enter to continue anyway (app will not work), or Ctrl+C to exit..."
fi

# Build frontend for production
echo "Building frontend..."
cd ../frontend
npm run build
cd ..

# Serve static files from backend
echo "Starting backend with static file serving..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

sleep 5

# Get the proxy URL
BASE_URL=$(echo $JUPYTERHUB_SERVICE_PREFIX | sed 's/\/$//')

echo ""
echo "========================================="
echo "✅ App is ready!"
echo "========================================="
echo "Open this URL in your browser:"
echo "https://hub.gesis.mybinder.org${BASE_URL}/proxy/8000/"
echo "========================================="
echo ""
echo "To stop the server: pkill -f uvicorn"
echo ""
