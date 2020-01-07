#!/usr/bin/env bash

# internal script for jenkins

renice -n 20 $$
source /net/software/python/2020_01/anaconda/bin/activate

# set a prefix for each cmd
green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

# Force Exit 0
# trap 'exit 0' EXIT SIGINT SIGTERM

# source internal_toolbox/bash/cuda.bash
git clone https://github.com/fgnt/paderbox

# include common stuff (installation of toolbox, paths, traps, nice level...)
source paderbox/jenkins_common.bash

# pip install --user -e toolbox/
pip install --user -e .[tests]

pytest --junitxml='test_results.xml' --cov=pb_bss  \
  --doctest-modules --doctest-continue-on-failure --cov-report term -v "tests/" || true # --processes=-1
# Use as many processes as you have cores: --processes=-1
# Acording to https://gist.github.com/hangtwenty/1aeb36ee85f4bdce0899
# `--cov-report term` solves the problem that doctests are not included
# in coverage

# Export coverage
python -m coverage xml --include="pb_bss*"

# Pylint tests
pylint --rcfile="paderbox/pylint.cfg" -f parseable pb_bss > pylint.txt || true
# --files-output=y is a bad option, because it produces hundreds of files

pip freeze > pip.txt
pip uninstall --quiet --yes pb_bss
