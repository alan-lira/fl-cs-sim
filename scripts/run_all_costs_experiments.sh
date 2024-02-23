#!/bin/bash
# Append the project's root directory to the 'PYTHONPATH' environment variable.
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd"
# Enable debugging for the current shell session.
set -x
# Find and execute all experiment files that end with '_costs_experiment.py'.
find ./experiments/ -name "*_costs_experiment.py" | while read line; do
    python3 $line
done
# Exit.
exit
