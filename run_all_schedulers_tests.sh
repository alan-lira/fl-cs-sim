#!/bin/bash
# Append the project's root directory to the 'PYTHONPATH' environment variable.
cwd=$(pwd)
export PYTHONPATH="${PYTHONPATH}:$cwd"
# Enable debugging for the current shell session.
set -x
# Find and execute all test files that start with 'test_' and end with '.py'.
find tests/ -name "test_*.py" | while read line; do
    python3 -m unittest $cwd/$line
done
# Exit.
exit
