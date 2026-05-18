# Use an official lightweight Python runtime as our base image
FROM python:3.10-slim

# Set environment variables to optimize Python for container environments:
# 1. PYTHONDONTWRITEBYTECODE=1: Prevents Python from writing .pyc files to disk.
# 2. PYTHONUNBUFFERED=1: Ensures Python outputs stdout and stderr immediately without buffering,
#    allowing SGT logs to be streamed and captured live in Docker/cloud environments.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Install system-level dependencies:
# - sqlite3: To query and manage our local reservation databases.
# - libgomp1: Essential parallel processing library required by LightGBM to run model calculations.
# --no-install-recommends and rm -rf /var/lib/apt/lists/* keep our image extremely compact and clean.
RUN apt-get update && \
    apt-get install -y --no-install-recommends sqlite3 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to take full advantage of Docker's layer caching.
# If requirements.txt doesn't change, Docker will skip installing dependencies on subsequent builds.
COPY requirements.txt /app/

# Install the project Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files and project directories from our workspace into the container
COPY . /app/

# Ensure our Linux pipeline shell script is fully executable inside the container
RUN chmod +x run.sh

# Set the default runtime command for our container:
# It runs generate_mock_db.py to create data/noshow.db, then executes run.sh to trigger the pipeline.
CMD ["sh", "-c", "python generate_mock_db.py && ./run.sh"]
