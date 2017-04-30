# -*- coding: utf-8 -*-

#make use of useful features which are independent
import numpy as np
import matplotlib.pyplot as plt


greyhounds = 500
labradors = 500


#take random height for population
grey_height = 28 + 4 * np.random.randn(greyhounds)
labs_height = 24 + 4 * np.random.randn(labradors)


plt.hist([grey_height, labs_height], stacked = 'True', color = ['r', 'b'])
plt.show()

