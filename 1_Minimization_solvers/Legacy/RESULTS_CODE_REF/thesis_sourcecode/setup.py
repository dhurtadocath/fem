from setuptools import setup, find_packages

__version__ = "0.0.1"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# with open("requirements.txt", "r") as f:
#     requirements = f.read().splitlines()

setup(
    # packages=find_packages(where="src", include=["model_training*"]),
    # package_dir={"": "src"},
    version=__version__
    # install_requires=requirements
)