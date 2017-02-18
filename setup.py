from setuptools import find_packages
from setuptools import setup


setup(
    name='tools',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'transfer-learn=tools.command:transfer_learn_command',
        ],
    },
)
