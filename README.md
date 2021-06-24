# READ ME

## Install

````shell script
cd TechLandscape
pyenv local 3.8.5  # Nb: in theory, any version of python 3.8 should do it.  
poetry env use path/to/python3.8  # Eg ~/.pyenv/shims/python3.8
poetry install
pre-commit install
````

See doc for:
 
- [poetry][doc-poetry] - *python package manager*
- [pre-commit][doc-pre-commit] - *framework for pre-commit hooks*

[doc-poetry]:"https://poetry.eustace.io/docs/"
[doc-pre-commit]:"https://pre-commit.com"

## Usage

At this stage, the easiest way to explore the project is to walk through `example.py`.
 
 