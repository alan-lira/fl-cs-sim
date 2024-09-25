#!/bin/bash

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Script started message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Enable debugging for the current shell session.
set -x

# Create the 'scripts_output' folder (if needed).
mkdir scripts_outputs

# Run the setup to install Python3 modules dependencies.
bash ./scripts/run_setup.sh 2>&1 | tee scripts_outputs/run_setup.out

# Run all schedulers' unit tests to verify their implementations are working correctly.
bash ./scripts/run_all_schedulers_tests.sh 2>&1 | tee scripts_outputs/run_all_schedulers_tests.out

# Run all costs experiments.
bash ./scripts/run_all_costs_experiments.sh 2>&1 | tee scripts_outputs/run_all_costs_experiments.out

# Run all timing experiments.
bash ./scripts/run_all_timing_experiments.sh 2>&1 | tee scripts_outputs/run_all_timing_experiments.out

# Run all experiments results analyses.
bash ./scripts/run_all_experiments_results_analyses.sh 2>&1 | tee scripts_outputs/run_all_experiments_results_analyses.out

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Exit.
exit 0
