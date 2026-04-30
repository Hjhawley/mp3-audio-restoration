#!/bin/bash

# let 'er rip
# automatically retries on error and resumes from last chunk

while true; do
    echo "===== Starting training chunk ====="
    python3 model_training.py
    if [ $? -ne 0 ]; then
        echo "!!! Script crashed. Retrying in 15 seconds..."
        sleep 15
    else
        echo "===== Chunk completed successfully. Continuing... ====="
    fi
done
