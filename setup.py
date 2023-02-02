from setuptools import find_packages
from setuptools import setup


setup(
    name="DLPlotTools",
    use_scm_version=True,
    packages=find_packages(exclude=['pToolTest.*']),
    install_requires=['dash==2.7.1',
                      'dash-bootstrap-components==1.3.0',
                      'dash-core-components==2.0.0',
                      'dash-html-components==2.0.0',
                      'numpy==1.24.1',
                      'plotly==5.13.0',
                      'python-box==6.0.0',
                      'seaborn==0.12.2',
                      'toml==0.10.2',
                      'torch==1.13.1'],
    test_suite='pToolTest',
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