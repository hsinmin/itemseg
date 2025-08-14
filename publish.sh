#!/bin/bash

rm dist/*
rm -rf src/utvenvitemseg
hatch version minor
python3 -m build
python3 -m twine upload --repository pypi --verbose dist/*
