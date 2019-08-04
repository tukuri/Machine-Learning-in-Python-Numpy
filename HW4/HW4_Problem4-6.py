#Sung Hoon Choi
#CS/CNS/EE156a HW4 Problem 4
import math
import random

# The target sine function.
def f(x):
    return math.sin(math.pi * x)

# By lecture note 3, we can find the slope that minimizes mean square error as:
# w = x_psuedo_inverse * y
# If we solve this matrix arithmetic for two point cases (x1,y1) (x2,y2), then we obtain
# regression_slope = (x1y1+x2y2)/(x1^2+x2^2)
def regression_slope(x1, f_x1, x2, f_x2):
    return ((x1*f_x1)+(x2*f_x2))/(x1**2+x2**2)   # slope for the linear equation that minimizes mean square error.

# return the corresponding g_bar(x) = ax
def g(slope, x):
    return slope * x

total_repetition = 10000
total_slope = 0

for i in range (0, total_repetition):
    x1 = (1 if random.random() < 0.5 else -1) * random.random() # x1 is a random value in [-1,1]
    x2 = (1 if random.random() < 0.5 else -1) * random.random() # x2 is a random value in [-1,1]
    f_x1 = f(x1)
    f_x2 = f(x2)
    current_slope = regression_slope(x1, f_x1, x2, f_x2)
    total_slope = total_slope + current_slope

g_bar_slope = total_slope/total_repetition
print ("g_bar's slope: ", g_bar_slope)  #Answer for Problem 4.

######################### Code for Problem 5 starts ############################

total_bias = 0
for i in range (0, total_repetition):
    x = (1 if random.random() < 0.5 else -1) * random.random() # x is a random value in [-1,1]
    current_bias = (g(g_bar_slope, x) - f(x))**2
    total_bias = total_bias + current_bias

print("bias: ", total_bias/total_repetition) # Answer for Problem 5

######################## Code for Problem 6 starts #############################

total_variance = 0
for i in range(0, total_repetition):
    x1 = (1 if random.random() < 0.5 else -1) * random.random()  # x1 is a random value in [-1,1]
    x2 = (1 if random.random() < 0.5 else -1) * random.random()  # x2 is a random value in [-1,1]
    f_x1 = f(x1)
    f_x2 = f(x2)
    current_slope = regression_slope(x1, f_x1, x2, f_x2)

    D_x = (1 if random.random() < 0.5 else -1) * random.random()  # x2 is a random value in [-1,1]
    current_variance = (g(current_slope, D_x) - g(g_bar_slope,D_x))**2
    total_variance = total_variance + current_variance

print("variance: ", total_variance/total_repetition) # Answer for Problem 6