# SpinModel2D
Python code for modelling a two-dimensional lattice of interacting spins. Can be used to model ferromagnetic and anti-ferromagnetic, single layer lattices.

The package uses mathematical optimisation software to create the models, expressing the physics problem in terms of an objective function to be minimised. The model of the spins follows closely to what the Classical Heisenberg model does.

The documentation for the code is contained in the Code_documentation.pdf file.

## Dependencies
* Python 2.7.
* A scientific distribution of Python (such as Anaconda) is recommended, else you will have to manually install NumPy, SciPy and matplotlib.
* Pyomo: This is an optimisation package used. Installation instructions are on the [Pyomo website](http://www.pyomo.org/installation/).
* Ipopt: This is the solver used by Pyomo. Installation instructions are on the [Ipopt website](http://www.coin-or.org/Ipopt/documentation/node10.html)
* pyOpt: Another optimisation package. Installation instructions are on the [pyOpt website](http://www.pyopt.org/install.html)

## Installation
Clone the repository. The code is the folder model2D, so ensure you have the location in your Python path in order to import from it. 

## Using the code
The documentation provides descriptions on how to use the code.

## Acknowledgments
* Pyomo modelling language
* Ipopt solver
* SciPy Python library
* pyOpt python package
* Diamond Light Source Ltd, where the code was written during the summer of 2016
