[![Build Status](https://api.travis-ci.org/Benardi/ml_algorithms.svg?branch=master)](https://travis-ci.org/Benardi/ml_algorithms)
[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Machine Learning Algorithms

Machine Learning algorithms implemented in Python

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
**Optionally**: venv

### Installing

Clone and enter the directory using cd

```
git clone https://github.com/Benardi/ml_algorithms

cd ml_algorithms 
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

To execute tests simply run 

```
tox
``` 

## Authors

* **Benardi Nunes** - *Initial work* - [Benardi](https://github.com/Benardi)

See also the list of [contributors](https://github.com/Benardi/ml_algorithms/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **johnthagen** - python-blueprint [example repo](https://github.com/johnthagen/python-blueprint)
