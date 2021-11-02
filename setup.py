from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ['copy']

setup(
    name="snspec",
    version="0.0.6",
    author="Jiawen Yang",
    author_email="jiawen.yang096@gmail.com",
    description="A package to analyze Type Ia supernova spectra.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/yang-jiawen/snspec",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
