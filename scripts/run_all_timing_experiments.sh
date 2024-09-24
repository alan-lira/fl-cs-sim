#!/bin/bash

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Script started message.
echo "The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Append the project's root directory to the 'PYTHONPATH' environment variable.
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd"

# Enable debugging for the current shell session.
set -x

# Find and execute all experiment files that end with '_timing_experiment.py'.
find ./experiments/ -name "*_timing_experiment.py" | while read -r line; do
    python3 "$line"
done

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "The '$script_file' script has ended successfully!"
echo "Elapsed time: $((end_time - start_time)) seconds."

# Exit.
exit 0
