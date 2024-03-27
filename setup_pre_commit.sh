#!/bin/bash
python3 -m pip install pre-commit
pre-commit install
python3 -m pip install ruff
python3 -m pip install black