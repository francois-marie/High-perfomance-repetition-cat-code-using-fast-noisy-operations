from setuptools import setup, find_packages

setup(
    name='qsim',
    packages=find_packages(),
    author='francois-marie',
    install_requires=['qutip', 'pymatching', 'pandas', 'seaborn']
)