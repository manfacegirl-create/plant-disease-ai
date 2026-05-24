#!/bin/bash
# Start Streamlit in background on port 8501, then Flask on port 5000

echo "Starting Streamlit ML scanner on port 8501..."
streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false &

STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID"

# Give Streamlit a moment to boot
sleep 3

echo "Starting Flask app on port 5000..."
STREAMLIT_URL="http://localhost:8501" python app.py
