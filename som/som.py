'''
Created on Mar 28, 2018

@author: 703188429
'''
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import math
from scipy.spatial.distance import cdist

class SOM():
    def __init__(self, neurons, dimentions, n_iter=1000, learning_rate=0.1):
        neighbourhood_radius = max(neurons) / 2
        self.neurons = neurons
        self.dimentions = dimentions
        self.weights = np.random.randint(0, 255, size=(neurons[0], neurons[1], dimentions)) / 255
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.initial_neighbourhood_radius = neighbourhood_radius
        self.neighbourhood_radius = neighbourhood_radius
        self.time_constant = n_iter/np.log(self.initial_neighbourhood_radius)
        self.fig = plt.figure()


    def train(self, samples):
        # Define Time constant used by learning rate and neighbourhood radius
        dimentions = self.weights.shape
        weightDiffences = []
        for i in range(1, self.n_iter+1):
            print("Iteration :", i)
            for _ in samples:
                sample = random.choice(samples)
                """For each sample check which neuron has closest proximity"""
                distances = cdist(self.weights.reshape(dimentions[0]*dimentions[1], dimentions[2]), sample, metric='euclidean')
                distances = distances.reshape(dimentions[0], dimentions[1])
                indices = np.where(distances == distances.min())  # Getting the neuron with min distance
                current_weights = self.weights
                self.updateWeights(i, sample, indices)

                weightDiff = np.sqrt(np.sum(np.multiply(self.weights - current_weights, self.weights - current_weights)))
                weightDiffences.append(weightDiff)
                #print("Iteration :", i, " Weight Diff:", weightDiff)
                #exit()

            if i % 1000 == 0:
                self.display(samples, 
                             "Iteration: " + str(i) + 
                             " | LR: %s %s" % (self.initial_learning_rate, self.learning_rate) +
                             " | NR: %s %s" % (self.initial_neighbourhood_radius, self.neighbourhood_radius))

            self.updateLearningRate(i)
            self.updateNeighbourhoodRadius(i)
            print("NR:", self.initial_neighbourhood_radius, self.neighbourhood_radius)
            print("LR:", self.initial_learning_rate, self.learning_rate)
            #print(self.weights)
            #break
        self.display(samples, "Final", show=True)
        self.displayWeightDiff(weightDiffences)

    def updateWeights(self, iteration, sample, indices):
        dimentions = self.weights.shape

        """ Caculate the neighbourhood neurons which will be selected for weight update"""
        influenceRegionX1 = max(0, indices[0][0] - self.neighbourhood_radius) 
        influenceRegionX2 = min(self.weights.shape[0], indices[0][0] + self.neighbourhood_radius) 
        influenceRegionY1 = max(0, indices[1][0] - self.neighbourhood_radius) 
        influenceRegionY2 = min(self.weights.shape[1], indices[1][0] + self.neighbourhood_radius) 
        influenceRegionX1 = int(np.round(influenceRegionX1))
        influenceRegionX2 = int(np.round(influenceRegionX2))
        influenceRegionY1 = int(np.round(influenceRegionY1))
        influenceRegionY2 = int(np.round(influenceRegionY2))
        influenceVector = np.zeros((self.weights.shape[0], self.weights.shape[1]))
        influenceVector[influenceRegionX1: influenceRegionX2, influenceRegionY1: influenceRegionY2] = 1
        influenceVector = influenceVector.reshape(self.weights.shape[0], self.weights.shape[1], 1)

        #print(influenceRegionX1, influenceRegionX2, influenceRegionY1, influenceRegionY2)
        
        """ Caculate how much each neighbourhood neurons will affect the weight basted on their distance"""
        closestNeuron = self.weights[indices[0][0], indices[1][0]]
        distances = cdist(self.weights.reshape(dimentions[0]*dimentions[1], dimentions[2]), 
                          closestNeuron.reshape(1, dimentions[2]), metric='euclidean')
        distances = distances.reshape(dimentions[0], dimentions[1])
        #print(indices)
        #print(self.weights)
        #print(self.weights[indices[0][0], indices[1][0]])
        influenceValues =  np.exp(-np.multiply(distances, distances) / (2 * self.neighbourhood_radius *  self.neighbourhood_radius * iteration))
        influenceValues = influenceValues.reshape(self.weights.shape[0], self.weights.shape[1], 1)
        
        influenceValues = np.multiply(influenceVector, influenceValues)

        """Update weights using learning rate, influence """
        #current_weights = self.weights
        #print("current_weights:", current_weights[0])
        #self.learning_rate = 3
        self.weights = self.weights + np.multiply(influenceValues, (sample - self.weights))  * self.learning_rate
        #print(np.multiply(influenceValues, (sample - self.weights))[0])
        #print(np.multiply(influenceValues, (sample - self.weights))[0] * self.learning_rate)
        #print("weights:", self.weights[0])

        #weightDiff = cdist(self.weights[0], current_weights[0], metric='euclidean')
        #d = np.sqrt(np.sum(np.multiply(self.weights[0] - current_weights[0], self.weights[0] - current_weights[0])))
        #print("Weight Diff:", weightDiff, d)

    def updateLearningRate(self, iteration):
        self.learning_rate = self.initial_learning_rate * np.exp(-iteration/self.n_iter)

    def updateNeighbourhoodRadius(self, iteration):
        self.neighbourhood_radius = self.initial_neighbourhood_radius * np.exp(-iteration/self.time_constant)

    def display(self, samples, title, show=False):
        dimentions = self.weights.shape
        if not show:
            plt.ion()
        ax = self.fig.add_subplot(111, projection='3d')
        plt.title(title)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        for sample in samples:
            ax.scatter(sample[0][0], sample[0][1], sample[0][2], c=sample, marker='.')
        for weight in self.weights.reshape(dimentions[0]*dimentions[1], dimentions[2]):
            ax.scatter(weight[0], weight[1], weight[2], c='black', marker='X')

        if show:
            plt.show()
        else:
            plt.pause(0.05)

    def displayWeightDiff(self, weightDiffences):
        plt.plot(weightDiffences)
        plt.ylabel('weightDiffences')
        plt.show()

def main():
    num_training = 400
    samples = []
    choices = [1, 5, 10, 90, 80, 85]
    for _ in range(num_training):
        sample = np.array([random.randint(1, 100) / float(100),
                        random.randint(1, 100) / float(100),
                        random.randint(1, 100) / float(100)])
        sample = np.array([random.choice(choices) / float(100),
                        random.choice(choices) / float(100),
                        random.choice(choices) / float(100)])
        sample = sample.reshape(1, 3)
        samples.append(sample)
    
    s = SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
    s.train(samples)
    exit()      
    fig = plt.figure()
    ax = fig.gca()
    x = 0
    y = 0
    gridsize = np.sqrt(num_training)
    for sample in samples:
        x += 1
        if x > gridsize:
            x = 0
            y += 1
        plt.scatter(x, y, c=sample)
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()