#!/bin/bash
# Executable bash script to execute the complete end-to-end hotel no-show ML pipeline.
# Do not install dependencies in this run.sh file as per requirements.

echo "=========================================="
echo "Executing End-to-End ML Pipeline (run.sh)"
echo "=========================================="

if command -v python3 &> /dev/null; then
    python3 main.py
else
    python main.py
fi

echo "=========================================="
echo "Pipeline execution finished."
echo "=========================================="
