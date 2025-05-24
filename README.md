![test-coverage](/media/coverage.svg)
[![DOI](https://zenodo.org/badge/898884500.svg)](https://doi.org/10.5281/zenodo.15492629)
[![Publish Python Package to PyPI](https://github.com/mathjoha/sca/actions/workflows/hatch-publish-to-pypi.yml/badge.svg)](https://github.com/mathjoha/sca/actions/workflows/hatch-publish-to-pypi.yml)
[![CI-tox](https://github.com/mathjoha/sca/actions/workflows/tox.yml/badge.svg)](https://github.com/mathjoha/sca/actions/workflows/tox.yml)
<a href="https://creativecommons.org/licenses/by-nc/4.0/"><img decoding="async" loading="eager" src="https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by-nc.png" width="71" height="25" align="right"></a>

# Structural Collocation Analysis

This is a python + sqlite implementation of the method
_Structural Collocation Analysis_ as described in
[Structural reading: Developing the method of Structural Collocation Analysis using a case study on parliamentary reporting](https://doi.org/10.1080/01615440.2024.2414259)
and used in
[Democracy (Not) on Display: A Structural Collocation Analysis of the Mother of All Parliaments’ Reluctance to Broadcast Herself](https://doi.org/10.1093/pa/gsad002)

## Installation & Usage

### User Installation

You can install the package directly from GitHub using pip:

```bash
python -m pip install git+https://github.com/matjoha/sca.git
```

```bash
python -m pip install scolan
```

### Developer Setup

If you want to contribute to the development of SCA, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/matjoha/sca.git
cd sca
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

3. Install the package in editable mode with development dependencies:
```bash
python -m pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
python -m pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```


### Running Tests

The project uses pytest for testing and maintains 100% code coverage. To run the tests:

```bash
pytest
```

To run tests with coverage report:
```bash
pytest --cov=src/sca tests/
```

To run tests across different Python versions using tox:
```bash
tox
```

### Code Quality

The project enforces code quality through:
- Black for code formatting
- isort for import sorting
- pytest for testing
- 100% test coverage requirement

These checks are automatically run through pre-commit hooks and CI/CD pipelines.

## Citing SCA

If you use Structural Collocation Analysis in your research, please cite the
following article:

```BibTeX
@article{Johansson02072024,
author = {Mathias Johansson and Betto van Waarden},
title = {Structural reading: Developing the method of Structural Collocation Analysis using a case study on parliamentary reporting},
journal = {Historical Methods: A Journal of Quantitative and Interdisciplinary History},
volume = {57},
number = {3},
pages = {185-198},
year = {2024},
publisher = {Routledge},
doi = {10.1080/01615440.2024.2414259},
URL = {https://doi.org/10.1080/01615440.2024.2414259},
eprint = {https://doi.org/10.1080/01615440.2024.2414259}
}
```


# License

The code is published under a Creative Commons Attribution-NonCommercial
4.0 International license [CC BY-NC 4.0 license](/LICENSE).
