from setuptools import setup

setup(
    name='aware_cf_calculator',
    version='1.0',
    description='package to calculate static and stochastic AWARE characterization factors',
    license="MIT",
    author='Pascal Lesage',
    author_email='pascal.lesage@polymtl.ca',
    url="https://github.com/pascal.lesage/aware_cf_calculator/",
    packages=['aware_cf_calculator'],
    install_requires=[
        'numpy',
        'stats_arrays',
        'pandas',
        'pathlib',
        'pyprind',
        'openpyxl',
        'xlrd',
        'setuptools',
        'seaborn'
    ],
)