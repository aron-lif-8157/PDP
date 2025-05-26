import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("results100000.csv", delimiter=",", usecols=0)
plt.hist(data, bins=20, edgecolor='black')
plt.xlabel("Susceptible Humans at T=100")
plt.ylabel("Frequency")
plt.title("Histogram of Susceptible Humans (N=100k)")
plt.show()