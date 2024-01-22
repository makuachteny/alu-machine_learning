#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
''' This module plots a histogram with a normal distribution'''


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.hist(student_grades, edgecolor='black', bins= 10)
plt.show()
