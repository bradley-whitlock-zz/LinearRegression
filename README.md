# Linear Regression

## Linear Regression with Gradient Descent Implementation

## This repo was made as an introduction to the concepts of machine learning

A python script to take input from CSV and find a relationship between data and yeild a plot

### Setup on a mac (with homebrew)
First install python
```
brew install python
```

Then open a new terminal, and install virtualenv and virtualenvwrapper to setup enviroment 
```
pip install virtualenv
pip install virtualenvwrapper
mkdir ~/.virtualenvs
```

Open a new terminal and go to this project directory.

### Building project (dependencies)
Install dependencies in an isolated environment like so:
```
virtualenv ~/.virtualenvs/grad_dec
source ~/.virtualenvs/grad_dec/bin/activate
./build.sh
```

### Executing script
```
python -m grad_dec.main
```
