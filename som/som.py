'''
Created on Mar 28, 2018

@author: 703188429
'''
import random
import time
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import numpy as np


class SOM():
    def __init__(self, neurons, dimentions, n_iter=1000, learning_rate=0.1):
        neighbourhood_radius = np.sum(neurons)
        #neighbourhood_radius = (neurons[0] + neurons[1] )/ 3
        #neighbourhood_radius = max(window_width, window_height) / 2

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
        #weightDiffences = []
        for i in range(1, self.n_iter+1):
            print("Iteration :", i)
            for _ in samples:
                sample = random.choice(samples)
                """For each sample check which neuron has closest proximity"""
                distances = cdist(self.weights.reshape(dimentions[0]*dimentions[1], dimentions[2]), sample, metric='euclidean')
                distances = distances.reshape(dimentions[0], dimentions[1])
                indices = np.where(distances == distances.min())  # Getting the neuron with min distance

                #current_weights = self.weights
                self.updateWeights(sample, indices)
                # Just to find the display metric
                #weightDiff = np.sqrt(np.sum(np.multiply(self.weights - current_weights, self.weights - current_weights)))
                #weightDiffences.append(weightDiff)

            self.updateLearningRate(i)
            self.updateNeighbourhoodRadius(i)
        #self.displayWeightDiff(weightDiffences)

    def updateWeights(self, sample, indices):
        dimentions = self.weights.shape
        
        """ Caculate how much each neighbourhood neurons will affect the weight basted on their distance"""
        closestNeuron = self.weights[indices[0][0], indices[1][0]]
        
        # Calcualte the euclidean distances of neurons from BMU
        distances = cdist(self.weights.reshape(dimentions[0] * dimentions[1], dimentions[2]), 
                          closestNeuron.reshape(1, dimentions[2]), metric='euclidean')
        # Arrange distances as sorted index
        distances = np.argsort(np.argsort(distances.reshape(dimentions[0] * dimentions[1])))
        distances = distances.reshape(dimentions[0], dimentions[1])
        
        # Mark the neurons with distance greater then neighbourhood_radius as with influence 0
        influenceVector = copy.deepcopy(distances)
        influenceVector[distances > self.neighbourhood_radius] = -1
        influenceVector[influenceVector >= 0] = 1
        influenceVector[influenceVector == -1] = 0

        influenceValues =  np.exp(-np.multiply(distances, distances) / (2 * self.neighbourhood_radius * self.neighbourhood_radius))
        influenceValues = np.multiply(influenceVector, influenceValues)
        influenceValues = influenceValues.reshape(self.weights.shape[0], self.weights.shape[1], 1)

        """Update weights using learning rate, influence """
        self.weights = self.weights + np.multiply(influenceValues, (sample - self.weights))  * self.learning_rate

    def updateLearningRate(self, iteration):
        """Function to update the learning rate"""
        self.learning_rate = self.initial_learning_rate * np.exp(-iteration/self.n_iter)

    def updateNeighbourhoodRadius(self, iteration):
        """Function to update the neighbourhood radius"""
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
            ax.scatter(weight[0], weight[1], weight[2], c='red', marker='X')

        if show:
            plt.show()
        else:
            plt.pause(0.05)

    def displayWeightDiff(self, weightDiffences):
        plt.plot(weightDiffences)
        plt.ylabel('weightDiffences')
        plt.show()

    def displayClusters(self, samples):
        plt.plot(samples)
        plt.ylabel('weightDiffences')
        plt.show()

def main():
    
    num_training = 100
    samples = []
    choices = [1, 5, 10, 90, 80, 85]
    
    for _ in range(num_training):
        #sample = np.array([random.randint(1, 100) / float(100),
        #                random.randint(1, 100) / float(100),
        #                random.randint(1, 100) / float(100)])
        sample = np.array([random.choice(choices) / float(100),
                        random.choice(choices) / float(100),
                        random.choice(choices) / float(100)])
        sample = sample.reshape(1, 3)
        samples.append(sample)
    
    s = SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
    s.train(samples)
    s.display(samples, "Final", show=True)
    s.displayClusters(samples)

if __name__ == "__main__":
    main()