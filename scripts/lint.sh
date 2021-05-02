#!/bin/bash

python3 -m black scalabel
python3 -m isort scalabel
python3 -m pylint scalabel
python3 -m pydocstyle --convention=google scalabel
python3 -m mypy --strict --show-error-codes scalabel
