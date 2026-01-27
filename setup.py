from typing import List
from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a requirements file and returns a list of dependencies.
    '''
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='mlproj',
    version='0.0.1',
    author='sriman',
    author_email='srimanchaudhuri@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)