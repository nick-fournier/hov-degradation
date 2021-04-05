#!/usr/bin/env python3
# flake8: noqa
"""Setup script for the hov-analysis repository."""
from os.path import dirname, realpath
from setuptools import find_packages, setup
from hov_degradation.version import __version__


def _read_requirements_file():
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='hov-analysis',
    version=__version__,
    packages=find_packages(),
    install_requires=_read_requirements_file(),
    description='hov-analysis: Modeling and analysis to detect anomalies in '
                'HOV stations in District 7 of California',
    author='Yashar Farid, modified by Nicholas Fournier',
    url='https://github.com/nick-fournier/hov-degradation',
    author_email='yzfarid@berkeley.edu and nick.fournier@berkeley.edu',
    zip_safe=False,
    entry_points={"console_scripts": ["hov-analysis=hov_degradation.__main__:main"]},
)
