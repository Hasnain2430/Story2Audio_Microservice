FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .
COPY xtts_model/ xtts_model/


# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# Accept Coqui TTS license
ENV COQUI_TOS_AGREED=true

# Copy and allow execution of start.sh
COPY start.sh start.sh
RUN chmod +x start.sh

# Run both gRPC server and Streamlit app
CMD ["./start.sh"]
