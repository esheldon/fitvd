import os
from glob import glob
from distutils.core import setup

scripts=glob('bin/*')
scripts = [s for s in scripts if '~' not in s]


setup(
    name="fitvd",
    version="v0.9.5",
    description="Code to fit models to objects in DES+VISTA using MOF",
    license = "GPL",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    scripts=scripts,
    packages=['fitvd'],
)
