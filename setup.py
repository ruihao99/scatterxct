import os
import setuptools

if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

setuptools.setup(
    name="scatterxct",
    version="0.0.1",
    author="Rui-Hao Bi",
    author_email="biruihao@westlake.edu.cn",
    description="A convenient package for Exact wavefunction solution to non-adiabatic scattering problem.",
    long_description="A convenient package for Exact wavefunction solution to non-adiabatic scattering problem.",
    packages=['scatterxct'],
    entry_points={
        'console_scripts': [
            'simulation_scripts = scatterxct.simulation_scripts:main',
        ]
    },
    requires=requirements,
)

