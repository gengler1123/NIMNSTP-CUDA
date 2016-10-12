# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:07:35 2016

@author: Gary
"""

import matplotlib.pyplot as plt
import csv

X = []
Y = []

with open('entire_network_distance.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        X.append(float(row[0]))
        Y.append(float(row[1]))
        
plt.figure()
plt.plot(X,Y,'.')
plt.show()