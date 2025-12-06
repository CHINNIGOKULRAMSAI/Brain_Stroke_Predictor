from setuptools import find_packages,setup
from typing import List

HYPEN_DOT_E = '-e .'
def get_requirements(file_path: str) -> List[str]:

    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [re.replace('\n','') for re in requirements]

        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)

    return requirements

setup(
    author='jarvis',
    version='0.0.0.1',
    name='Brain_Stroke_Prediction',
    author_email='chinnigokul43@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)