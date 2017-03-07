from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os 
import csv
from helper import (max_multi_dim_array, min_multi_dim_array, avg_array, avg_multi_dim_array, graph)

dir_path = os.path.dirname(os.path.realpath(__file__))

def line_error(theta, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        x = x_data[i][1]
        y = y_data[i]
        totalError += (y - hyp(theta, x)) ** 2
    return totalError / float(len(x_data))

def hyp(theta, x):
    return theta[0] * x + theta[1]

def step_gradient(theta, x_data, y_data, learningRate): 
    t_gradient = [0] * len(theta)
    N = float(len(x_data))
    for i in range(0, len(x_data)):
        x = x_data[i][1]
        y = y_data[i]
        t_gradient[0] += x * (hyp(theta, x) - y)
        t_gradient[1] += (hyp(theta, x) - y)
    theta[1] = theta[1] - (learningRate * t_gradient[0])/N
    theta[0] = theta[0] - (learningRate * t_gradient[1])/N
    return theta

def gradient_descent_runner(x, y, theta, learning_rate, num_iterations):
    last_error = line_error(theta, x, y)
    count = 0
    for i in range(num_iterations):
        theta = step_gradient(theta, x, y, learning_rate)
        curr_error = line_error(theta, x, y)
        if last_error < curr_error:
            break
        last_error = curr_error
        count += 1
    print "Number of itterations: ", count
    return theta

def float_wrapper(reader):
    for v in reader:
        # yeild used like return, returns generator (itterate once)
        yield map(float, v)

# Improvment: use std. dev. rather than range
def feature_scale(data, index):
    x_vals = map(lambda x: x[index], data)
    std_dev = np.std(x_vals)
    #range_data = max_multi_dim_array(points, index) - min_multi_dim_array(points, index)
    #print range_data
    average_data = avg_multi_dim_array(data, index)
    for i in range(0, len(data)):
        data[i][index] = float(data[i][index] - average_data) / std_dev
    return data

def populate(file_name):
    X = []
    Y = []
    points = []
    with open(dir_path + '/' + file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        reader = float_wrapper(reader)
        for row in reader:
            X.append([1])
            X[len(X) - 1] += row[0:len(row) - 1]
            Y.append(row[len(row) - 1])
            points.append(row)
    return (X, Y)

def run():
    X, Y = populate('housing_data.csv')
    # initial guess for the coefficients
    theta = [0] * len(X[0])
    learning_rate = 0.0005
    max_num_iterations = 2000

    print X
    #X = feature_scale(X, 1)
    print X

    print "Running..."

    theta = gradient_descent_runner(X, Y, theta, learning_rate, max_num_iterations)
    x_plot = map(lambda x: x[1], X)
    print x_plot
    plt.plot(x_plot, Y, 'ro')
    x = np.array(range(0, int(max(x_plot) * 1.15)))
    formula = str(str(theta[1]) + '+' + str(theta[0]) + '*x')
    print formula   
    y = eval(formula)
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    run()

# I have noticed that as the "X" data gets smaller need to do 1
#   of 2 things to improve performance
# 1. increase learning learning
# 2. increase the number of itterations
#   - Also noticed that small change in learning rate dramatically
#       reduces the number of necessary itterations
# This is because with smaller changes(errors) the change each 
#   itteration is smaller