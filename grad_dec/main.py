from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os 
import csv
from helper import (max_multi_dim_array, min_multi_dim_array, avg_array, avg_multi_dim_array, graph)

dir_path = os.path.dirname(os.path.realpath(__file__))

def line_error(theta, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        totalError += (y - hyp(theta, x)) ** 2
    return totalError / float(len(points))

def hyp(theta, x):
    return theta[0] * x + theta[1]

def step_gradient(theta, points, learningRate):  
    t_gradient = [0] * len(theta)
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        t_gradient[0] += x * (hyp(theta, x) - y)
        t_gradient[1] += (hyp(theta, x) - y)
    theta[1] = theta[1] - (learningRate * t_gradient[0])/N
    theta[0] = theta[0] - (learningRate * t_gradient[1])/N
    return theta

def gradient_descent_runner(points, theta, learning_rate, num_iterations):
    for i in range(num_iterations):
        temp_theta = step_gradient(theta, array(points), learning_rate)
    return temp_theta

def float_wrapper(reader):
    for v in reader:
        # yeild used like return, returns generator (itterate once)
        yield map(float, v)

# Improvment: use std. dev. rather than range
def feature_scale(points, index):
    range_data = max_multi_dim_array(points, index) - min_multi_dim_array(points, index)
    average_data = avg_multi_dim_array(points, index)
    for i in range(0, len(points)):
        points[i][index] = float(points[i][index] - average_data) / range_data
    return points

def populate(file_name):
    #X = []
    #Y = []
    points = []
    with open(dir_path + '/' + file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        reader = float_wrapper(reader)
        for row in reader:
            #X.append([1])
            #X[len(X) - 1] += row[0:len(row) - 1]
            #Y.append(row[len(row) - 1])
            points.append(row)
    return points


def run():
    points = populate('housing_data.csv')
    # initial guess for the coefficients
    theta = [0] * len(points[0])
    learning_rate = 0.0003
    num_iterations = 5000
    #print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    theta = gradient_descent_runner(points, theta, learning_rate, num_iterations)
    #print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

    X = []
    Y = []
    for row in points:
        X.append(row[0])
        Y.append(row[1])
    plt.plot(X, Y, 'ro')
    x = np.array(range(-2, 2))
    formula = str(str(theta[1]) + '+' + str(theta[0]) + '*x')   
    print formula
    y = eval(formula)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    run()
