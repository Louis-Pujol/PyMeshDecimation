# This script is used to lint/build/test the package locally.

black .
pip uninstall pymeshdecimation -y
python -m build
pip install dist/pymeshdecimation-0.0.2-cp38-cp38-linux_x86_64.whl
pytest .