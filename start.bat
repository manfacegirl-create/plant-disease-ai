@echo off
echo Starting LeafSentry...
echo.

echo [1/2] Starting Streamlit ML scanner on port 8501...
start "Streamlit" cmd /k "streamlit run streamlit_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false"

echo Waiting for Streamlit to boot...
timeout /t 4 /nobreak > nul

echo [2/2] Starting Flask website on port 5000...
set STREAMLIT_URL=http://localhost:8501
start "Flask" cmd /k "python app.py"

timeout /t 2 /nobreak > nul
echo.
echo Done! Open your browser at: http://localhost:5000
echo.
start http://localhost:5000
