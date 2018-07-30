'''
Created on Apr 9, 2018

@author: 703188429
'''

class SOM():
    def __init__(self, neurons, dimentions, n_iter=1000, learning_rate=0.1, mode="numpy"):
        self.obj = None
        if mode == "numpy":
            from somlib.som_numpy import SOM_NUMPY
            self.obj = SOM_NUMPY(neurons, dimentions, n_iter, learning_rate)
        elif mode == "tensor":
            from somlib.som_tensor import SOM_TENSOR
            self.obj = SOM_TENSOR(neurons, dimentions, n_iter, learning_rate)  
          
        self.predict = self.obj.predict
        self.displayClusters = self.obj.displayClusters
        self.labels_ = self.obj.labels_
        self.weights_ = self.obj.weights_

    def train(self, samples):
        self.train = self.obj.train(samples)
        self.labels_ = self.obj.labels_
        self.weights_ = self.obj.weights_

    def main(self):
        print("OLS")

if __name__ == "__main__":
    import random
    import numpy as np
    num_training = 200
    samples = []
    choices = [1, 5, 10, 90, 80, 85]
    for _ in range(num_training):
        sample = np.array([random.randint(1, 100) / float(100),
                        random.randint(1, 100) / float(100),
                        random.randint(1, 100) / float(100)])
#         sample = np.array([random.choice(choices) / float(100),
#                         random.choice(choices) / float(100),
#                         random.choice(choices) / float(100)])
        sample = sample.reshape(1, 3)
        samples.append(sample)

    import time
    starttime = time.time()
    s = SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1, mode="tensor")
    s.train(samples)
    endtime = time.time() - starttime
    print("Total TIME: ", endtime)
    print("Cluster centres:", s.weights_)
    print("labels:", s.labels_)
    result = s.predict(samples)
    print(result)
    #s.displayClusters(samples)
