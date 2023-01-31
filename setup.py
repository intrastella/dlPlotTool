from setuptools import find_packages
from setuptools import setup


setup(
    name="DLPlotTools",
    use_scm_version=True,
    packages=find_packages(),
    classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python :: 3.10',
        ],

    include_package_data=True,
    author="Stella Muamba Ngufulu",
    author_email="stellamuambangufulu@gmail.com",
    description="Ligand prediction repo for experimental purposes.",
    license="Mozilla Public License 2.0",
    platforms="linux")