# Optimize not mecessary
def max_array(arr):
	i = 0
	maximum = 0
	while i < len(arr):
		if arr[i] >= arr[maximum]:
			maximum = i
		i += 1
	return arr[maximum]

# optimize not necessary - min(list)
def min_array(arr):
	i = 0
	minimum = 0
	while i < len(arr):
		if arr[i] <= arr[minimum]:
			minimum = i
		i += 1
	return arr[minimum]

def max_multi_dim_array(arr, index):
	i = 0
	maximum = 0
	while i < len(arr):
		if arr[i][index] >= arr[maximum][index]:
			maximum = i
		i += 1
	return arr[maximum][index]

def min_multi_dim_array(arr, index):
	i = 0
	minimum = 0
	while i < len(arr):
		if arr[i][index] <= arr[minimum][index]:
			minimum = i
		i += 1
	return arr[minimum][index]

def avg_array(arr):
	i = 0
	total = 0
	while i < len(arr):
		total += arr[i]
		i += 1
	return total/len(arr)

def avg_multi_dim_array(arr, index):
	i = 0
	total = 0
	while i < len(arr):
		total += arr[i][index]
		i += 1
	return total/len(arr)

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    plt.show()