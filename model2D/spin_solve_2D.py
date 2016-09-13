'Python script for creating a 2D model of interacting spins'


import pyomo.environ
from pyomo.opt import SolverFactory
import time
import scipy.optimize
import numpy as np
import pyOpt
from spin_plot import plot_spins


class Vector(object):
    'Class for a 2D vector in Cartesian space'
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _init_spins_rule(pyomo_model, i, j):
    'Returns the spin (i,j) in the initial configuration'
    return pyomo_model.start_spins[i, j]


def _get_bound_coord(input_coord, problem):
    'Gets the coordinate relating input_coord depending on boundary conditions'
    if input_coord.i in range(problem.row_no) and\
       input_coord.j in range(problem.col_no):
        return input_coord
    elif problem.boundary == 'zero':
        return None  # Indicates there is "no spin" here
    elif problem.boundary == 'periodic':

        # Matches the point back into the defined grid
        if input_coord.i < 0:
            while input_coord.i not in range(problem.row_no):
                input_coord.i += problem.row_no
        elif input_coord.i > problem.row_no - 1:
            while input_coord.i not in range(problem.row_no):
                input_coord.i -= problem.row_no

        if input_coord.j < 0:
            while input_coord.j not in range(problem.col_no):
                input_coord.j += problem.col_no
        elif input_coord.j > problem.col_no - 1:
            while input_coord.j not in range(problem.row_no):
                input_coord.j -= problem.col_no
        return input_coord


def _get_spin(input_coord, problem, is_pyomo, spin_array):
    '''Returns the spin at input_coord.
    If is_pyomo is True, it returns the Pyomo spin variable.
    If is_pyomo is False, then it will get the spin from spin_array which
    is a 2D numpy array'''
    bound_coord = _get_bound_coord(input_coord, problem)
    if is_pyomo:
        return problem.model.s[bound_coord.i, bound_coord.j]
    else:
        return spin_array[bound_coord.i, bound_coord.j]


def _adj_sum(input_coord, problem, is_pyomo, spin_array):
    '''Calculates the sum of the scalar products of the adjacent spins to the
    spin at input_coord. If is_pyomo is True it returns this as the sum of the
    pyomo variables, else it returns the floatof the sum from the numpy array
    containg spins'''
    input_spin = _get_spin(input_coord, problem, is_pyomo, spin_array)
    # List of the spin values in the adjacent coordinates to input_coord
    adj_spin_list = [_get_spin(Coord(input_coord.i-1, input_coord.j), problem,
                               is_pyomo, spin_array),
                     _get_spin(Coord(input_coord.i+1, input_coord.j), problem,
                               is_pyomo, spin_array),
                     _get_spin(Coord(input_coord.i, input_coord.j-1), problem,
                               is_pyomo, spin_array),
                     _get_spin(Coord(input_coord.i, input_coord.j+1), problem,
                               is_pyomo, spin_array)]

    # Sum of the scalar product of spins
    if is_pyomo:
        return sum(pyomo.environ.cos(input_spin - adj_spin)
                   for adj_spin in adj_spin_list)
    else:
        return sum(np.cos(input_spin - adj_spin)
                   for adj_spin in adj_spin_list)


def _diag_sum(input_coord, problem, is_pyomo, spin_array):
    '''Calculates the sum of the scalar products of the closest diagonal spins
    to the spin at input_coord. If is_pyomo is True it returns this as the sum
    of the pyomo variables, else it returns the float of the sum from the numpy
    array containing spins'''
    input_spin = _get_spin(input_coord, problem, is_pyomo, spin_array)
    # List of the spin values in the closest diagonal coords to the input_coord
    diag_spin_list = [_get_spin(Coord(input_coord.i-1, input_coord.j-1),
                                problem, is_pyomo, spin_array),
                      _get_spin(Coord(input_coord.i+1, input_coord.j-1),
                                problem, is_pyomo, spin_array),
                      _get_spin(Coord(input_coord.i-1, input_coord.j+1),
                                problem, is_pyomo, spin_array),
                      _get_spin(Coord(input_coord.i+1, input_coord.j+1),
                                problem, is_pyomo, spin_array)]
    # Sum of the scalar product of spins
    if is_pyomo:
        return sum(pyomo.environ.cos(input_spin - diag_spin)
                   for diag_spin in diag_spin_list)
    else:
        return sum(np.cos(input_spin - diag_spin)
                   for diag_spin in diag_spin_list)


def _python_hamiltonian(problem, spin_array):
    '''Returns the Hamiltonian using standard Python code (as opposed to Pyomo
    objects), applied to the given numpy array: spin_array'''
    tot_adj = 0.0  # Total of the adjacent scalar products
    tot_diag = 0.0  # Total of the diagonal scalar products
    tot_mag = 0.0  # Total of the magnetic field interaction terms
    tot_aniso = 0.0  # Total of the terms from single-ion-anisotropy

    is_pyomo = False  # Indicates this is not for Pyomo

    for input_coord in problem.coord_array:
        # Loops over all the coordinates in the grid and adds to totals
        input_spin = _get_spin(input_coord, problem, is_pyomo, spin_array)
        tot_adj += _adj_sum(input_coord, problem, is_pyomo, spin_array)
        tot_diag += _diag_sum(input_coord, problem, is_pyomo, spin_array)
        tot_mag += np.cos(problem.mag_ang - input_spin)
        tot_aniso += (np.sin(problem.aniso_ang - input_spin))**2

    return -problem.adj_coup*tot_adj - problem.diag_coup*tot_diag \
        - problem.mag_strength*tot_mag \
        + problem.aniso_strength*tot_aniso


class Coord(object):
    'Class for a coordinate object i.e. i - row number, j - column number'
    def __init__(self, i, j):
        self.i = i
        self.j = j


class SpinProblem(object):
    '''Class which takes in the parameters of an interacting spin problem and
    solves it with a choice of solver'''
    def __init__(self, grid_dimen, boundary="periodic", start_spins=None,
                 bas_vec1=(1.0, 0.0), bas_vec2=(0.0, 1.0), adj_coup=1.0,
                 diag_coup=0.0, aniso=(0.0, 0.0), mag_fiel=(0.0, 0.0)):
        '''
        Input explanation:
        * grid_dimen  - Tuple containing dimensions of grid.
        I.e. (m,n) -> m rows, n columns
        * boundary    - Boundary conditions of the grid. Either "periodic"
        or "zero"
        * start_spins - Initial condition for the spins. If None, then all
        spins will be 0.0. Else they are given in a numpy array
        * bas_vec1    - First basis vector of the lattice given as a tuple
        * bas_vec2    - Second basis vector of the lattive given as a tuple
        * adj_coup    - Coupling term between adjacent neighbouring spins
        * diag_coup    - Coupling term between spins on the closest
        * aniso       - Tuple representing single-ion-anisotropy.
        First term is strength, second is angle (in radians) that it points in
        * mag_fiel    - Tuple representing magnetic field. First term is
        strength,second is angle (in radians) that it points in
        '''
        # ___Stores the input parameters as class variables___

        self.row_no = grid_dimen[0]
        self.col_no = grid_dimen[1]
        self.boundary = boundary

        if start_spins is None:
            self.start_spins = np.zeros((self.row_no, self.col_no))
        else:
            self.start_spins = start_spins

        self.bas_vec1 = Vector(x=bas_vec1[0], y=bas_vec1[1])
        self.bas_vec2 = Vector(x=bas_vec2[0], y=bas_vec2[1])

        self.adj_coup = adj_coup
        self.diag_coup = diag_coup

        self.aniso_strength = aniso[0]
        self.aniso_ang = aniso[1]

        self.mag_strength = mag_fiel[0]
        self.mag_ang = mag_fiel[1]

        # Variables to store the solver data we want
        self.results_array = None  # Array to store optimised spins
        self.solve_time = None  # Time the solver took to solve the  problem
        self.result_obj = None  # The resulting objective function

        # Derived variables
        self.no_spins = self.row_no * self.col_no  # Total number of spins
        # Array to hold then Coord objects in for the grid points
        coord_array_2d = np.array([[Coord(i, j) for j in range(self.col_no)]
                                   for i in range(self.row_no)])
        self.coord_array = coord_array_2d.flatten()

    def scipy_solve(self, solver):
        '''Solves the problem using a SciPy solver specified by the input,
        where "solver" is a string, i.e. solver = "Nelder-Mead"'''
        # Clearing the solution variables
        self.results_array = None
        self.result_obj = None
        self.solve_time = None

        start_spins_1d = self.start_spins.flatten()  # Array of intial spins

        solver_output = None  # Variable that stores the solver output
        time_1, time_2 = 0.0, 0.0  # Variables used for timing the solvers

        def _1d_hamiltonian(spin_array_1d):
            '''Gives a 1D numpy array of the spins as inputs and returns the
            Hamiltonian'''
            spin_array_2d = spin_array_1d.reshape((self.row_no, self.col_no))
            return _python_hamiltonian(self, spin_array_2d)

        if solver == "Diff_Evo":
            # Differential evolution requires different conditions
            bounds = [(0, 2*np.pi) for i in range(self.no_spins)]

            time_1 = time.time()
            solver_output = scipy.optimize.differential_evolution(
                _1d_hamiltonian, bounds)
            time_2 = time.time()
        else:
            # Any optimize.minimize solver
            time_1 = time.time()
            solver_output = scipy.optimize.minimize(
                _1d_hamiltonian, start_spins_1d, method=solver)
            time_2 = time.time()

        # Spin results
        results_1d = solver_output.x
        self.results_array = results_1d.reshape((self.row_no, self.col_no))

        # Objective function
        self.result_obj = solver_output.fun

        # Solve time
        self.solve_time = time_2 - time_1

    def pyopt_solve(self, solver):
        '''Solves the problem using a pyOpt solver specified by the input,
        where "solver" is a pyOpt object, i.e. solver = pyOpt.SOLVOPT()'''
        # Clearing the solution variables
        self.results_array = None
        self.result_obj = None
        self.solve_time = None

        def opt_func(spin_vars):
            'The optimisation function used by pyOpt'
            # List of the variables used for spins
            spin_list = [spin_vars[i] for i in range(self.no_spins)]

            # 2D numpy array for the spin list
            spin_array = np.asarray(spin_list).reshape((self.row_no,
                                                        self.col_no))

            obj_fun = _python_hamiltonian(self, spin_array)  # Objective func

            constr_list = []  # List of constraints)

            fail = 0  # Indicates the success of the solver

            return obj_fun, constr_list, fail

        # Sets up the pyOpt optimization
        opt_prob = pyOpt.Optimization('Spin system', opt_func)

        for input_coord in self.coord_array:
            # Adds the variables for the spins as pyOpt variables
            spin_name = 's[' + str(input_coord.i) + ', ' \
                    + str(input_coord.j) + ']'
            this_start_spin = self.start_spins[input_coord.i, input_coord.j]
            opt_prob.addVar(spin_name, lower=0.0, upper=2*np.pi,
                            value=this_start_spin)

        opt_prob.addObj('obj_fun')  # Tells pyOpt what the objective function

        solver(opt_prob)  # Tells the solver to solve the optimization problem

        # Set of solutions for the spin
        var_set = opt_prob.solution(0).getVarSet()
        results_list = [var_set[i].value for i in range(self.no_spins)]
        self.results_array = np.asarray(results_list).reshape((self.row_no,
                                                               self.col_no))

        # Objective function
        pyopt_result_obj = opt_prob.solution(0).getObj(0)
        self.result_obj = pyopt_result_obj.value

        # Solve time
        opt_prob.solution(0).write2file('pyOpt_output.txt')  # Output file

        # Searches the output file for the time measured by pyOpt
        for line in open('pyOpt_output.txt').readlines():
            if line.find('Total Time:') != -1:
                line_split = line.split()
                self.solve_time = line_split[line_split.index("Time:") + 1]
                break

    def pyomo_solve(self, solver):
        '''Solves the problem using a Pyomo solver specified by the input,
        where "solver" is a string, i.e. solver = "ipopt"'''
        # Clearing the solution variables
        self.results_array = None
        self.result_obj = None
        self.solve_time = None

        # Setting up the Pyomo model object
        self.model = pyomo.environ.ConcreteModel()

        # Range sets for the grid points
        self.model.row_points = pyomo.environ.RangeSet(0, self.row_no - 1)
        self.model.col_points = pyomo.environ.RangeSet(0, self.col_no - 1)

        # Initialisation of the spins
        self.model.start_spins = self.start_spins

        # Pyomo variable for the spins
        self.model.s = pyomo.environ.Var(self.model.row_points,
                                         self.model.col_points,
                                         domain=pyomo.environ.NonNegativeReals,
                                         initialize=_init_spins_rule)

        def _pyomo_hamil(pyomo_model):
            'The Hamiltonian function using the pyomo objects'
            tot_adj = 0.0  # Total of the adjacent scalar products
            tot_diag = 0.0  # Total of the diagonal scalar products
            tot_mag = 0.0  # Total of the magnetic field interaction terms
            tot_aniso = 0.0  # Total of the terms from single-ion-anisotropy

            for input_coord in self.coord_array:
                input_spin = _get_spin(input_coord, self, is_pyomo=True,
                                       spin_array=None)
                tot_adj += _adj_sum(input_coord, self, is_pyomo=True,
                                    spin_array=None)
                tot_diag += _diag_sum(input_coord, self, is_pyomo=True,
                                      spin_array=None)
                tot_mag += pyomo.environ.cos(self.mag_ang - input_spin)
                tot_aniso += (pyomo.environ.sin(self.aniso_ang -
                                                input_spin)) ** 2

            return -self.adj_coup*tot_adj - self.diag_coup*tot_diag - \
                self.mag_strength*tot_mag + self.aniso_strength*tot_aniso

        # Objective function (the Hamiltonian)
        self.model.OBJ = pyomo.environ.Objective(rule=_pyomo_hamil,
                                                 sense=pyomo.environ.minimize)

        # Solves the model
        opt = SolverFactory(solver)
        opt.solve(self.model, logfile=solver+'.txt')

        # Creates the results array
        self.results_array = np.zeros((self.row_no, self.col_no))
        for i in self.model.row_points:
            for j in self.model.col_points:
                self.results_array[i, j] = \
                                    pyomo.environ.value(self.model.s[i, j])

        self.result_obj = pyomo.environ.value(self.model.OBJ)
        # Searches the log file to find the solving time
        line_str = "Total CPU secs in IPOPT (w/o function evaluations)   ="
        if solver == 'ipopt':
            for line in reversed(open(solver+'.txt').readlines()):
                if line.find(line_str) != -1:
                    line_split = line.split()
                    self.solve_time = line_split[line_split.index("=") + 1]
                    break

    def plot(self, plot_name=None):
        'Plots the results of the optimization problem'
        # Lists that store the x and y positions in real space of the spins
        x_pos, y_pos = [], []
        # Lists that store the components of the vectors of the spins
        x_spin_comps, y_spin_comps = [], []

        for spin_coord in self.coord_array:
            # Loops over the spins and fills the lists
            x_pos.append(spin_coord.i * self.bas_vec1.x +
                         spin_coord.j * self.bas_vec2.x)
            y_pos.append(spin_coord.i * self.bas_vec1.y +
                         spin_coord.j * self.bas_vec2.y)
            x_spin_comps.append(np.cos(self.results_array[spin_coord.i,
                                                          spin_coord.j]))
            y_spin_comps.append(np.sin(self.results_array[spin_coord.i,
                                                          spin_coord.j]))

        plot_spins((x_pos, y_pos, x_spin_comps, y_spin_comps))
