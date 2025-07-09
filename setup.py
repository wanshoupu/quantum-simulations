from setuptools import setup, find_packages

# Load requirements from file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="quompiler",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
)
