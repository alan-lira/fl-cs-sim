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

# Enable debugging for the current shell session.
set -x

# Set the virtual environment path.
venv_path="$script_parent_dir/.venv_fl_cs_sim"

# Create the virtual environment.
python3 -m venv "$venv_path"

# Activate the virtual environment.
source "$venv_path"/bin/activate

# Create the 'scripts_output' folder (if needed).
scripts_output_folder="$script_parent_dir"/scripts_outputs
mkdir "$scripts_output_folder"

# Run the setup to install Python3 modules dependencies.
bash "$script_work_dir"/run_setup.sh 2>&1 | tee "$scripts_output_folder"/run_setup.out

# Run all schedulers' unit tests to verify their implementations are working correctly.
bash "$script_work_dir"/run_all_schedulers_tests.sh 2>&1 | tee "$scripts_output_folder"/run_all_schedulers_tests.out

# Run all costs experiments.
bash "$script_work_dir"/run_all_costs_experiments.sh 2>&1 | tee "$scripts_output_folder"/run_all_costs_experiments.out

# Run all timing experiments.
bash "$script_work_dir"/run_all_timing_experiments.sh 2>&1 | tee "$scripts_output_folder"/run_all_timing_experiments.out

# Run all experiments results analyses.
bash "$script_work_dir"/run_all_experiments_results_analyses.sh 2>&1 | tee "$scripts_output_folder"/run_all_experiments_results_analyses.out

# Deactivate the virtual environment.
deactivate

# Remove the virtual environment.
rm -rf "$venv_path"

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Exit.
exit 0
