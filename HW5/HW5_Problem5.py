#Sung Hoon Choi
#CS/CNS/EE156a HW5 Problem 5 and Problem 6
import math
import random

def Error(u,v):
    return (u*math.exp(v)-2*v*math.exp(-u))**2

def gradient_u(u,v):
    return 2*(u*math.exp(v)-2*v*math.exp(-u))*(math.exp(v)+2*v*math.exp(-u))

def gradient_v(u,v):
    return 2*(u*math.exp(v)-2*v*math.exp(-u))*(u*math.exp(v)-2*math.exp(-u))

u=1
v=1
iteration = 0
learning_rate = 0.1
Max_iteration=50
initial_error = Error(u,v)

while (Error(u,v)>10**(-14)):
    iteration = iteration +1
    grad_u=gradient_u(u,v)
    grad_v=gradient_v(u,v)
    u = u - learning_rate*grad_u
    v = v - learning_rate*grad_v
    print("iteration: %d u: %f v: %f" %(iteration,u,v))

print("Taken iterations: %d u: %f v:%f" %(iteration,u,v)) #Answer for problem 5 and 6


