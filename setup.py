from setuptools import setup
from typing import List

REQUIREMENTS_FILENAME = 'requirements.txt'


def get_requirements_list()->List[str]:
    """
    This function is going to return list of requirements present in requirements.txt file

    returns a list of all library names needed to be installed to run the app.
    """
    with open(REQUIREMENTS_FILENAME, 'r') as requirements_file:
        return requirements_file.readlines()

setup(
    name = 'housing-price-predictor',
    version='0.0.1',
    author='Abhinesh Kourav',
    description='This is the first FSDS ML project',
    packages=['housing'],
    install_requires = get_requirements_list()
)