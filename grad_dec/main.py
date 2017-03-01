import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
from helper import (max_multi_dim_array, min_multi_dim_array, avg_array, avg_multi_dim_array, graph)
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))

# Number of dimensions to yield "Y"
num_dimensions = 1

# Training Set, X : Training data, Y : actual data
X = []
Xraw = []
Y = []

# Coefficients (Theta), init to 1
T = [-1] * (num_dimensions + 1)
# Aplha
A = 0.005
# Number of training examples
M = 0

def int_wrapper(reader):
    for v in reader:
    	# yeild used like return, returns generator (itterate once)
        yield map(int, v)

def hypothesis(training_num): 
	hyp = 0
	itt = 0
	for i in X[training_num]:
		hyp += (i * T[itt])
	hyp += T[0]
	return hyp

def update_t(index):
	total = 0
	i = 0
	while i < M:
		if index == 0:
			#print 'Index = 0, i: ', i, ' Total: ', hypothesis(i),  Y[i]
			total += (hypothesis(i) - Y[i])
		else:
			total += (hypothesis(i) - Y[i]) * X[i][index - 1]
		i += 1
	total = total * A / M
	return (T[index] - total)

def gradient_descent():
	i = 0
	tempT = [0] * (num_dimensions + 1)
	while i < len(T):
		tempT[i] = update_t(i)
		i += 1
	return tempT

def cost():
	total = 0
	i = 0
	while i < M:
		total += (hypothesis(i) - Y[i]) ** 2
		i += 1
	return (total / (2 * (M + 1)))

# Improvment: use std. dev. rather than range
def feature_scale(arr, index):
	i = 0
	range_data = max_multi_dim_array(arr, index) - min_multi_dim_array(arr, index)
	average_data = avg_multi_dim_array(arr, index)
	while i < M:
		arr[i][index] = float(arr[i][index] - average_data) / range_data
		i += 1
	return arr

def prediction(sq_feet):
	# Must scale the sq_feet first
	range_data = max_multi_dim_array(Xraw, 0) - min_multi_dim_array(Xraw, 0)
	average_data = avg_multi_dim_array(Xraw, 0)
	sq_feet = float(sq_feet - average_data) / range_data
	return (T[0] + T[1]*sq_feet)


# Format of CSV file is (square ft, value)
with open(dir_path + '/housing_data_exact.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter= ',')
	reader = int_wrapper(reader)

	for row in reader:
		X.append(row[0:len(row) - 1])
		Xraw.append(row[0:len(row) - 1])
		Y.append(row[len(row) - 1])
		M += 1

# Feature Scaling
X = feature_scale(X, 0)

# Gradient Descent
num_itt = 1000
costValues = []
i = 0

last_cost = cost()
print T
while i < num_itt:
	#Xplot.append(T[0])
	#Yplot.append(T[1])
	T = gradient_descent()
	cost_curr = cost()
	print "Diff: ", last_cost - cost_curr, " Cost_Curr: ", cost_curr
	if last_cost < cost_curr:
		break 
	last_cost = cost_curr
	costValues.append(cost_curr)
	#costValues.insert(0, cost())
	i += 1

print 'Min:', min(costValues)
print 'i: ', i
print 'Final T values: ', T
temp = prediction(2000)
print "prediction for 2000: ", temp
temp = prediction(3000)
print "prediction for 3000: ", temp
temp = prediction(4000)
print "prediction for 4000: ", temp


'''
# Plot of the cost function
plt.plot(costValues,'ro')
plt.axis([0,  num_itt + num_itt/10, min_array(costValues) - min_array(costValues) / 10, max_array(costValues) + max_array(costValues) / 10])
plt.show()
'''
# Plot of the data and line of best fit
Xplot = []
i = 0
while i < len(X):
	Xplot.append(X[i][0])
	i += 1


plt.plot(Xplot, Y, 'ro')
print min(Y)
x = np.array(range(-1, 1))
formula = str(T[0]) + '+' + str(T[1]) + '*x'
print formula
y = eval(formula)
plt.plot(x, y)
plt.axis([min(Xplot) - min(Xplot) / 10,  max(Xplot) + max(Xplot) / 10, 0 , max(Y) + max(Y) / 10])
plt.show()

