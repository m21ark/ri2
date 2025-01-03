#!/bin/bash

# Function to handle termination
cleanup() {
    kill -TERM -- -$$ 2>/dev/null
    exit 0
}

# Trap SIGINT and SIGTERM to clean up
trap cleanup INT TERM

# Start the first command
rcssserver3d &
echo "Started rcssserver3d with PID $!"

# Start the second command
sh src/robotViz/bin/roboviz.sh &
echo "Started roboviz.sh with PID $!"

# # Start the third command
# (cd src && python3 main.py) &
# echo "Started main.py with PID $!"

# Wait indefinitely to keep the parent process alive
echo "Parent process running. Press Ctrl+C to terminate."
wait

kill -9 $(ps -e | grep rcssserver3d | awk '{print $1}')
