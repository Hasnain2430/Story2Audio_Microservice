#!/bin/bash

# Start gRPC server in background
python server_ms.py &

# Start Streamlit frontend
streamlit run streamlit_ms.py
