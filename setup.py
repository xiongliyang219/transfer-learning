from setuptools import find_packages
from setuptools import setup


setup(
    name='tools',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'train=tools.command:train_command',
        ],
    },
)
