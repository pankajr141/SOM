'''
Created on Mar 28, 2018

@author: 703188429
'''
import random
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import numpy as np


class SOM():
    def __init__(self, neurons, dimentions, n_iter=1000, learning_rate=0.1):
        
        ##INITIALIZE GRAPH
        #self._graph = tf.Graph()
        #with self._graph.as_default():
        neighbourhood_radius = np.sum(neurons) * 1.0
        self.neurons = neurons
        self.dimentions = dimentions
        self.weights = tf.Variable(tf.random_uniform([neurons[0], neurons[1], dimentions], minval=0, maxval=1.0), tf.float32)
        self.initial_learning_rate = tf.constant(learning_rate * 1.0, dtype=tf.float32)
        self.learning_rate = tf.Variable(learning_rate * 1.0, dtype=tf.float32)
        self.n_iter = n_iter
        self.initial_neighbourhood_radius = tf.constant(neighbourhood_radius, dtype=tf.float32)
        self.neighbourhood_radius = tf.Variable(neighbourhood_radius, dtype=tf.float32)
        self.time_constant = tf.constant(n_iter/np.log(neighbourhood_radius), tf.float32)
        self.fig = plt.figure()
        self.sample = tf.placeholder(tf.float32, [1, dimentions], name='sample')
        self.weights_ = None
        self.ZERO = tf.constant(0.0)
        self.iteration= tf.Variable(0.0, tf.float32)
        self.defineUpdateWeightsGraph()
        self.defineUpdateLearningRateGraph()
        self.defineNeighbourhoodRadiusGraph()
        ##INITIALIZE SESSION
        self.sess = tf.Session()

        ##INITIALIZE VARIABLES
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
 
    def train(self, samples):

        for i in range(1, self.n_iter+1):
            print("Iteration :", i)
            for _ in samples:
                sample = random.choice(samples)
                #print(self.sess.run(self.distances, feed_dict={self.sample: sample}))
                #print(self.sess.run(self.influenceVector, feed_dict={self.sample: sample}))
                #print(self.sess.run(self.influenceValues, feed_dict={self.sample: sample}))
                """For each sample check which neuron has closest proximity"""
                self.sess.run(self._training_op, feed_dict={self.sample: sample})

                #self.updateWeights(sample) # Tremendously slow, as in case of  function call tensorflow is creating new variables.
            #self.sess.run(self.learning_rate, feed_dict={self.iteration:  float(i)})
            #self.sess.run(self.neighbourhood_radius, feed_dict={self.iteration:  float(i)})
            self.sess.run(self._training_op_lr, feed_dict={self.iteration:  float(i)})
            self.sess.run(self._training_op_nr, feed_dict={self.iteration:  float(i)})

            #self.updateLearningRate(i)
            #self.updateNeighbourhoodRadius(i)
#             if i % 10 == 0:
#                 self.weights_ = self.sess.run(self.weights)
#                 self.display(samples, 
#                             "Iteration: " + str(i) + 
#                             " | LR: %s %s" % (self.sess.run(self.initial_learning_rate), self.sess.run(self.learning_rate)) +
#                             " | NR: %s %s" % (self.sess.run(self.initial_neighbourhood_radius), self.sess.run(self.neighbourhood_radius)))

        self.weights_ = self.sess.run(self.weights)
    
    def defineUpdateWeightsGraph(self):
        dimentions = self.weights.shape
        weight_ = tf.reshape(self.weights, [dimentions[0]*dimentions[1], dimentions[2]])
        distances = tf.reduce_sum(tf.square(weight_ - self.sample), axis=1, keepdims=True)
        distances = tf.reshape(distances, (dimentions[0], dimentions[1]))
        indices = tf.where(distances <= tf.reduce_min(distances)) # Getting the neuron with min distance

        """ Caculate how much each neighbourhood neurons will affect the weight basted on their distance"""
        closestNeuron = tf.reshape(self.weights[indices[0][0], indices[0][1]], [1, dimentions[2]])

        weight_ = tf.reshape(self.weights, [dimentions[0]*dimentions[1], dimentions[2]])

        distances = tf.reduce_sum(tf.square(weight_ - closestNeuron), axis=1, keepdims=True)
        distances = tf.reshape(distances, (dimentions[0], dimentions[1]))

        # Arrange distances as sorted index        
        distances = tf.reshape(distances, [dimentions[0] * dimentions[1]])
        distances = tf.nn.top_k(distances, k=dimentions[0]*dimentions[1]).indices
        distances = distances[::-1]

        distances = tf.nn.top_k(distances, k=dimentions[0]*dimentions[1], sorted=True).indices
        distances = distances[::-1]
        distances = tf.reshape(distances, [dimentions[0], dimentions[1]])     
        distances = tf.cast(distances, tf.float32)

        # Mark the neurons with distance greater then neighbourhood_radius as with influence 0        
        influenceVector = tf.identity(distances)

        condition = tf.equal(influenceVector, self.ZERO)
        influenceVector = tf.where(condition, tf.ones_like(influenceVector, tf.float32), influenceVector)      
        condition = tf.greater(influenceVector, self.neighbourhood_radius)
        influenceVector = tf.where(condition, tf.zeros_like(influenceVector, tf.float32), influenceVector)
        condition = tf.greater(influenceVector, self.ZERO)
        influenceVector = tf.where(condition, tf.ones_like(influenceVector, tf.float32), influenceVector)

        influenceValues = tf.exp(-tf.multiply(distances, distances) / 
                                 (tf.constant(2.0) * tf.multiply(self.neighbourhood_radius, self.neighbourhood_radius)))        

        influenceVector = tf.cast(influenceVector, tf.float32)
        influenceValues = tf.cast(influenceValues, tf.float32)

        influenceValues = tf.multiply(influenceVector, influenceValues)

        #self.distances = distances
        #self.influenceVector = influenceVector
        #self.influenceValues = influenceValues

        influenceValues = tf.reshape(influenceValues, [self.weights.shape[0], self.weights.shape[1], 1])

        """Update weights using learning rate, influence """
        weights_new = self.weights + tf.multiply(influenceValues, (self.sample - self.weights))  * self.learning_rate
        self._training_op = tf.assign(self.weights, weights_new)  

    # Ideally this should be also definded like updateWeights fn since here tensorflow will create a new copy, but since these are very few in size we can ifnore for now
    def defineUpdateLearningRateGraph(self):
        """Function to update the learning rate"""
        learning_rate = self.initial_learning_rate * tf.exp(-self.iteration/self.n_iter)
        self._training_op_lr = tf.assign(self.learning_rate, learning_rate)
        
    def defineNeighbourhoodRadiusGraph(self):
        """Function to update the neighbourhood radius"""
        neighbourhood_radius = self.initial_neighbourhood_radius * tf.exp(-self.iteration/self.time_constant)
        self._training_op_nr = tf.assign(self.neighbourhood_radius, neighbourhood_radius)

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

        for weight in self.weights_.reshape(dimentions[0]*dimentions[1], dimentions[2]):
            ax.scatter(weight[0], weight[1], weight[2], c='red', marker='X')

        if show:
            plt.show()
        else:
            plt.pause(0.05)

    def displayWeightDiff(self, weightDiffences):
        plt.plot(weightDiffences)
        plt.ylabel('weightDiffences')
        plt.show()

def main():    
    num_training = 200
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
    
    import time
    starttime = time.time()
    s = SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
    s.train(samples)
    endtime = time.time() - starttime
    print("Total TIME: ", endtime)

    s.display(samples, "Final", show=True)
    #s.displayClusters(samples)

if __name__ == "__main__":
    main()