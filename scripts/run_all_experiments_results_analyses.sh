#!/bin/bash

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Script started message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Append the project's root directory to the 'PYTHONPATH' environment variable.
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd"

# Enable debugging for the current shell session.
set -x

# Find and execute all experiment analysis files that end with '_experiment_analysis.py'.
find ./analyses/ -name "*_experiment_analysis.py" | while read -r line; do
    python3 "$line"
done

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Exit.
exit 0
