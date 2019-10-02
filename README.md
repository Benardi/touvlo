[![Build Status](https://api.travis-ci.org/Benardi/touvlo.svg?branch=master)](https://travis-ci.org/Benardi/touvlo)
[![Documentation Status](https://readthedocs.org/projects/touvlo/badge/?version=latest)](https://touvlo.readthedocs.io/en/latest/?badge=latest)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Touvlo

This project provides Machine Learning algorithms and models implemented from scratch. These implementation aren't meant to be performatic, but instead to expose the logic of the components/blocks that make the Machine Learning models possible. For this reason the routines employed by the models are also provided and tested separately.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

To make use of this project you need both python3 and pip3.
Both are readily available in packages: 

```
sudo apt update
sudo apt install python3
sudo apt install python3-pip
```
To run the testing environments we have provided you'll also need to install tox

```
sudo apt update
sudo apt install tox
```

**Optionally**: venv

### Installing

Clone and enter the directory using cd

```
git clone https://github.com/Benardi/touvlo

cd touvlo 
```

Use venv to keep dependencies tidy, but you may opt not to use it.
Create a new directory inside the project directory where will keep the dependencies as 'venv'.

```
python3 -m venv ./venv
```

Source the venv to activate it.

```
source venv/bin/activate
```

Use pip to install the requirements

```
pip3 install -r requirements.txt
```

# Running the tests

To execute all testing environments simply run 

```
tox
``` 

## Unit tests

To execute only the unit tests, run 

```
tox -e py35
``` 

## Coding style tests

To execute only the coding style tests, run 

```
tox -e pep8
``` 

## Contributing

Please read [CONTRIBUTING.md](https://github.com/Benardi/touvlo/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/Benardi/touvlo/tags). 

## Authors

* **Benardi Nunes** - *Initial work* - [Benardi](https://github.com/Benardi)

See also the list of [contributors](https://github.com/Benardi/touvlo/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **johnthagen** - python-blueprint [example repo](https://github.com/johnthagen/python-blueprint)
