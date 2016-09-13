'Functions for plotting a 2D lattice of interacting spins'

import csv
import math
import matplotlib.pyplot as plt


def plot_spins(input_list, plot_name=None):
    '''Plots the configuration of spins, where input_list =
    (x_pos, y_pos, x_spin_comps, y_spin_comps), where x_pos and y_pos are
    arrays indicating the points where the spins are, and x_spin_comps and
    y_spin_comps are the components of the vectors of the spins at
    those points'''
    x_pos, y_pos = input_list[0], input_list[1]
    x_spin_comps, y_spin_comps = input_list[2], input_list[3]

    plt.figure()
    axes = plt.gca()
    plt.gca().set_aspect('equal', adjustable='box')

    axes.quiver(x_pos, y_pos, x_spin_comps, y_spin_comps)
    axes.plot(x_pos, y_pos, 'or')
    axes.set_xlim([min(x_pos) - 1, max(x_pos) + 1])
    axes.set_ylim([min(y_pos) - 1, max(y_pos) + 1])
    axes.set_xticks([])
    axes.set_yticks([])

    if plot_name is None:
        plt.draw()
        plt.show()
    else:
        plt.title(plot_name)
        plt.draw()
        fig_name = plot_name + ".png"
        plt.savefig(fig_name)
        plt.show()


def plot_from_file(filename, plot_name=None, bas_vec_1=(1.0, 0.0),
                   bas_vec_2=(0.0, 1.0)):
    '''Plots spins given from bas_vec_1 csv file. Default is square lattice'''
    grid = []
    with open(filename, 'r') as data:
        rows = csv.reader(data)
        grid = [[float(cell) for cell in row] for row in rows]

    x_pos, y_pos, x_spin_comps, y_spin_comps = [], [], [], []
    row_no = len(grid)
    col_no = len(grid[0])
    for i in range(row_no):
        for j in range(col_no):
            x_pos.append(i*bas_vec_1[0] + j*bas_vec_2[0])
            y_pos.append(i*bas_vec_1[1] + j*bas_vec_2[1])
            x_spin_comps.append(math.cos(grid[i][j]))
            y_spin_comps.append(math.sin(grid[i][j]))

    plot_spins((x_pos, y_pos, x_spin_comps, y_spin_comps), plot_name)

if __name__ == '__main__':
	plot_from_file('/home/thy68636/Documents/Results_Simple_ferromagnetic/Program_Outputs/First_test_solve/Test_Output/Test_4/results/problem_ID_5/Nelder-Mead.csv')
				   

