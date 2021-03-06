# SOM
This is python implementation for Kohonen Self Organizing map using numpy and tensor

## Installtion

**Python 3**
`pip install somlib`

## Usage

1. Numpy implementation

```
	from somlib import som
	s = som.SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
	s.train(samples)  # samples is a n x 3 matrix
	print("Cluster centres:", s.weights_)
	print("labels:", s.labels_)
	result = s.predict(samples)
```

Here 5,5 is the dimention of neurons, 3 is the number of features. samples is numpy array with each sample a 3 dimentional vector

2. Tensor implementation

```
	from somlib import som
	s = SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1, mode="tensor")
	s.train(samples)  # samples is a n x 3 matrix
	print("Cluster centres:", s.weights_)
	print("labels:", s.labels_)
	result = s.predict(samples)
```

### Display clusters
To display clusters after training use this

```s.displayClusters(samples)```


![clusters](https://image.ibb.co/hS4uCH/figure_3.png "Clusters")