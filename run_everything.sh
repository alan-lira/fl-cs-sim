#!/bin/bash
# Enable debugging for the current shell session.
set -x
# Install python3 modules dependencies.
bash ./setup.sh
# Run all schedulers' unit tests to verify their implementations are working correctly.
bash ./run_all_schedulers_tests.sh
# Run all costs experiments.
bash ./run_all_costs_experiments.sh
# Run all timing experiments.
bash ./run_all_timing_experiments.sh
# Run analysis of experiments' results.
bash ./run_analysis_of_experiments_results.sh
# Exit.
exit
