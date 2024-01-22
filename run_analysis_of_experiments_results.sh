#!/bin/bash
# Append the project's root directory to the 'PYTHONPATH' environment variable.
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd"
# Enable debugging for the current shell session.
set -x
# Find and execute all experiment analysis files that end with '_experiment_analysis.py'.
find ./experiments_analyzes/ -name "*_experiment_analysis.py" | while read line; do
    python3 $line
done
# Exit.
exit
