# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:07:35 2016

@author: Gary
"""

import matplotlib.pyplot as plt
import csv


with open('entire_network_distance.csv','rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        print(row)