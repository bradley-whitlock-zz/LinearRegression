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
        x = x_data[i]
        y = y_data[i]
        totalError += (y - hyp(theta, x)) ** 2
    return totalError / float(len(x_data))

# Accepts x which is array of training data so can customize the form
def hyp(theta, x):
    #print x
    # Form y = mx + b
    return theta[0] * x[1] + theta[1] * x[2] + theta[2]

def step_gradient(theta, x_data, y_data, learningRate, num_factors): 
    t_gradient = 0
    temp_theta = [0] * num_factors
    N = float(len(x_data))
    for j in range(0, num_factors):
        for i in range(0, len(x_data)):
            t_gradient += x_data[i][j] * (hyp(theta, x_data[i]) - y_data[i])
        temp_theta[j] = theta[j] - (learningRate * t_gradient)/N
    t_gradient = 0 
    return temp_theta

def gradient_descent_runner(x, y, theta, learning_rate, max_num_iterations, num_factors):
    last_error = line_error(theta, x, y)
    count = 0
    for i in range(max_num_iterations):
        theta = step_gradient(theta, x, y, learning_rate, num_factors)
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

def feature_scale(data, index):
    x_vals = map(lambda x: x[index], data)
    std_dev = np.std(x_vals)
    average_data = avg_multi_dim_array(data, index)
    for i in range(0, len(data)):
        data[i][index] = float(data[i][index] - average_data) / std_dev
    return data

def populate(file_name):
    X = []
    Y = []
    with open(dir_path + '/' + file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter= ',')
        reader = float_wrapper(reader)
        for row in reader:
            # This is for completeness, first factor is always 1
            X.append([1])
            # This is where the data is inputted from CSV
            X[len(X) - 1] += row[0:len(row) - 1]
            Y.append(row[len(row) - 1])
    return (X, Y)

def run():
    X, Y = populate('housing_data_2.csv')

    # num_factors will always be 1 greater than the actual number of factors becasue of constant coeff
    num_factors = 3
    theta = [0] * (num_factors)

    # Coeff's user sets
    learning_rate = 0.001
    max_num_iterations = 10000

    print X
    #X = feature_scale(X, 1)
    #print X

    print "Running..."
    theta = gradient_descent_runner(X, Y, theta, learning_rate, max_num_iterations, num_factors)
    

    x_plot = map(lambda x: x[1], X)
    print x_plot
    #print 'Max xplot: ', max(x_plot)
    plt.plot(x_plot, Y, 'ro')
    x = np.array(range(0, int(max(x_plot) * 1.15)))
    formula = str(str(theta[2]) + ' + ' + str(theta[1]) + ' * x + ' + str(theta[0]) + ' * x')
    print formula   
    y = eval(formula)
    #plt.axis([-2, 2, 0, 250])
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

# Works with this data in csv

# Next step is to add another factor


'''
-2,20
-1,15
0,10
1,5
2,0
3,-5
4,-10
'''