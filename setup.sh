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

# Start the third command
(cd src && python3 Run_One_vs_One.py) &
echo "Started Run_One_vs_One.py with PID $!"

# Wait indefinitely to keep the parent process alive
echo "Parent process running. Press Ctrl+C to terminate."
wait

