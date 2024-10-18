#!/bin/bash

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Get the script working directory.
script_work_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Get the script parent directory.
script_parent_dir="$(dirname "$script_work_dir")"

# Script started message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Append the project's root directory to the 'PYTHONPATH' environment variable.
export PYTHONPATH="${PYTHONPATH}:$script_parent_dir"

# Enable debugging for the current shell session.
set -x

# Find and execute all experiment files that end with '_costs_experiment.py'.
find "$script_parent_dir"/experiments -name "*_costs_experiment.py" | while read -r line; do
    python3 "$line"
done

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Exit.
exit 0
