from pathlib import Path

from setuptools import find_packages, setup

setup(
    name=Path(__file__).resolve().parent.name,
    packages=find_packages(),
)
