#!/bin/bash
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
# Exit.
exit
