'''
Created on Jul 30, 2018

@author: 703188429
'''

from somlib import som
import random
import numpy as np

num_training = 200
samples = []
choices = [1, 5, 10, 90, 80, 85]
for _ in range(num_training):
    sample = np.array([random.randint(1, 100) / float(100),
                    random.randint(1, 100) / float(100),
                    random.randint(1, 100) / float(100)])
    sample = sample.reshape(1, 3)
    samples.append(sample)

s = som.SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
s.train(samples)  # samples is a n x 3 matrix
print("Cluster centres:", s.weights_)
print("labels:", s.labels_)
result = s.predict(samples)
s.displayClusters(samples)
