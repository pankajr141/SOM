'''
Created on Mar 28, 2018

@author: 703188429
'''

import random
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd

class SOM_NUMPY():
    """
    The numpy based Kohonen self organizing map implementation.
    """
    def __init__(self, neurons, dimentions, n_iter=1000, learning_rate=0.1):
        neighbourhood_radius = np.sum(neurons)
        self.neurons = neurons
        self.dimentions = dimentions
        self.weights = np.random.randint(0, 255, size=(neurons[0], neurons[1], dimentions)) / 255
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.initial_neighbourhood_radius = neighbourhood_radius
        self.neighbourhood_radius = neighbourhood_radius
        self.time_constant = n_iter/np.log(self.initial_neighbourhood_radius)
        self.weights_ = None  # Cluster centres
        self.labels_ = None # Assign labels

#         self.fig = plt.figure()
    
    
    def _assignLabels(self, samples):
        dimentions = self.weights.shape
        self.weights_ = self.weights.reshape(dimentions[0] * dimentions[1], dimentions[2])
        labels = []
        for sample in samples:
            distances = cdist(self.weights_, sample, metric='euclidean')
            indices = np.where(distances == distances.min())
            labels.append(indices[0][0])
        self.labels_ = labels
    
    def _updateWeights(self, sample):
        dimentions = self.weights.shape
        
        # Fund BMU neuron
        distances = cdist(self.weights.reshape(dimentions[0]*dimentions[1], dimentions[2]), sample, metric='euclidean')
        distances = distances.reshape(dimentions[0], dimentions[1])
        indices = np.where(distances == distances.min())  # Getting the neuron with min distance
        
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

    def _updateLearningRate(self, iteration):
        """Function to update the learning rate"""
        self.learning_rate = self.initial_learning_rate * np.exp(-iteration/self.n_iter)

    def _updateNeighbourhoodRadius(self, iteration):
        """Function to update the neighbourhood radius"""
        self.neighbourhood_radius = self.initial_neighbourhood_radius * np.exp(-iteration/self.time_constant)

    def train(self, samples):
        # Define Time constant used by learning rate and neighbourhood radius
        for i in range(1, self.n_iter+1):
            print("Iteration :", i)
            for _ in samples:
                sample = random.choice(samples)
                """For each sample check which neuron has closest proximity"""
                #current_weights = self.weights
                self._updateWeights(sample)
                # Just to find the display metric
                #weightDiff = np.sqrt(np.sum(np.multiply(self.weights - current_weights, self.weights - current_weights)))
                #weightDiffences.append(weightDiff)
#             if i % 10 == 0:
#                 self.display(samples, 
#                              "Iteration: " + str(i) + 
#                              " | LR: %s %s" % (self.initial_learning_rate, self.learning_rate) +
#                              " | NR: %s %s" % (self.initial_neighbourhood_radius, self.neighbourhood_radius))
            self._updateLearningRate(i)
            self._updateNeighbourhoodRadius(i)
        self._assignLabels(samples)

        #self.displayWeightDiff(weightDiffences)
    
    def predict(self, samples):
        result = []
        for sample in samples:
            distances = cdist(self.weights_, sample, metric='euclidean')
            indices = np.where(distances == distances.min())  # Getting the neuron with min distance
            #print(indices[0][0])
            result.append(indices[0][0])
        return np.array(result)
                    

#     def display(self, samples, title, show=False):
#         dimentions = self.weights.shape
#         if not show:
#             plt.ion()
# 
#         ax = self.fig.add_subplot(111, projection='3d')
#         plt.title(title)
#         ax.set_xlabel('X Label')
#         ax.set_ylabel('Y Label')
#         ax.set_zlabel('Z Label')
# 
#         for sample in samples:
#             ax.scatter(sample[0][0], sample[0][1], sample[0][2], c=sample, marker='.')
#         for weight in self.weights.reshape(dimentions[0]*dimentions[1], dimentions[2]):
#             ax.scatter(weight[0], weight[1], weight[2], c='red', marker='X')
# 
#         if show:
#             plt.show()
#         else:
#             plt.pause(0.05)
# 
#     def displayWeightDiff(self, weightDiffences):
#         plt.plot(weightDiffences)
#         plt.ylabel('weightDiffences')
#         plt.show()

    def displayClusters(self, samples):
        from sklearn.decomposition import PCA
        samples = np.array(samples)
        samples = samples.reshape(samples.shape[0], samples.shape[2])
        df = pd.DataFrame(self.labels_, columns=["labels"])
        df['data'] = df.apply(lambda x: list(samples[x.name]), axis=1)
        
        plt.style.use('seaborn-white')
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        fig, ax = plt.subplots(self.weights.shape[0], self.weights.shape[1], sharex='col', sharey='row')
        cntr = -1
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                cntr += 1
                df_ = df[df['labels'] == cntr]
                print("Label:", cntr, "shape:", df_.shape)
                if df_.shape[0] <= 1:
                    continue 
                pca = PCA(n_components=2)
                samples_ = pca.fit_transform(df_['data'].tolist())
                for li, sample in enumerate(df_['data']):
                    ax[i, j].scatter(samples_[li][0], samples_[li][1], c=sample, marker='.')
                #ax[i, j].text(0.5, 0.5, str((2, 3, i)),
                #         fontsize=18, ha='center')
        plt.show()

def main():
    num_training = 5000
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
    s = SOM_NUMPY(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
    s.train(samples)
    endtime = time.time() - starttime
    print("Total TIME: ", endtime)
    print("Cluster centres:", s.weights_)
    print("labels:",s.labels_)
    result = s.predict(samples)
    print(result)
    #s.display(samples, "Final", show=True)    
    s.displayClusters(samples)

if __name__ == "__main__":
    main()
