# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:20:23 2021

@author: devanshi shah
"""

#importing numpy
import numpy as np
from matplotlib import pyplot as plt

#seeting the seed with the last 2 digits of student number
seed = 69 # Student last 2 digits
np.random.seed(seed)

# generating the values of uniform distribution by using random and storing in variable x.
x = np.random.uniform(size=100, low=-1, high=1)
print(x)

#relationship of y = 12x -4
y = 12 * x - 4
print(y)

#using matplotlib as plt now we create a scatter graph of x and y.
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('y = 12 * x - 4')
plt.savefig('part_1_scatter_plot_1')
plt.show()

# Add( injecting noise) to the y data using the from the normal (Gaussian) distribution.
noise = np.random.normal(size=100)

y += noise

plt.scatter(x,y, alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.title('y including noise')
plt.savefig('part_1_scatter_plot_2')

