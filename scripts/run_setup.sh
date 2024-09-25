#!/bin/bash

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Script started message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Install the Python3 modules dependencies through pip using the 'requirements.txt' file.
python3 -m pip install -r requirements.txt

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Exit.
exit 0
