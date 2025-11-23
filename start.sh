#!/bin/bash

# Start the FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Wait for backend to start
sleep 5

# Detect if running in MyBinder and set API URL accordingly
# In MyBinder, we need to use the proxy URL structure
if [ -n "$JUPYTERHUB_SERVICE_PREFIX" ]; then
    # Extract base URL and construct proxy URLs
    # The URL pattern is: /user/{username}/proxy/{port}/
    # Remove trailing slash if present
    BASE_URL=$(echo $JUPYTERHUB_SERVICE_PREFIX | sed 's/\/$//')
    
    # Set backend API URL for frontend
    export VITE_API_URL="${BASE_URL}/proxy/8000"
    
    # Set base path for Vite assets (without double slash)
    export VITE_BASE_PATH="${BASE_URL}/proxy/5173/"
    
    # Set HMR host for hot module replacement
    export VITE_HMR_HOST="hub.gesis.mybinder.org"
    
    echo ""
    echo "========================================="
    echo "🚀 MyBinder URLs"
    echo "========================================="
    echo "Frontend: https://hub.gesis.mybinder.org${BASE_URL}/proxy/5173/"
    echo "Backend:  https://hub.gesis.mybinder.org${BASE_URL}/proxy/8000/"
    echo "========================================="
    echo ""
    echo "Copy the Frontend URL above and paste it in your browser!"
    echo ""
else
    # Running locally
    echo "Local environment - using localhost"
    echo "Frontend: http://localhost:5173/"
    echo "Backend:  http://localhost:8000/"
fi

# Start the frontend dev server
echo "Starting React frontend..."
cd frontend
npm run dev -- --host 0.0.0.0 --port 5173

# Keep the script running
wait
