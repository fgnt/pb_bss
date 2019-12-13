#!/usr/bin/env bash

# internal script for jenkins

renice -n 20 $$
source /net/software/python/2018_12/anaconda/bin/activate

# set a prefix for each cmd
green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

# Force Exit 0
# trap 'exit 0' EXIT SIGINT SIGTERM

# Use a pseudo virtualenv, http://stackoverflow.com/questions/2915471/install-a-python-package-into-a-different-directory-using-pip
mkdir -p venv
export PYTHONUSERBASE=$(readlink -m venv)

# source internal_toolbox/bash/cuda.bash

# pip install --user -e toolbox/
pip install -e .[tests]

pytest --junitxml='test_results.xml' --cov=pb_bss  \
  --doctest-modules --doctest-continue-on-failure --cov-report term -v "tests/" || true # --processes=-1
# Use as many processes as you have cores: --processes=-1
# Acording to https://gist.github.com/hangtwenty/1aeb36ee85f4bdce0899
# `--cov-report term` solves the problem that doctests are not included
# in coverage

# Export coverage
python -m coverage xml --include="pb_bss*"

# Pylint tests
# pylint --rcfile="pylint.cfg" -f parseable pb_bss > pylint.txt || true
# --files-output=y is a bad option, because it produces hundreds of files

pip freeze > pip.txt
pip uninstall --quiet --yes pb_bss
