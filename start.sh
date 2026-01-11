#!/bin/sh
#start fastapi
echo "Starting FastAPI..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

#start streamlit application
echo "Starting Streamlit..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo "Streamlit exited"