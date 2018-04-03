# SOM
This is python implementation for Kohonen Self Organizing map using numpy and tensor

## Usage

1. Numpy implementation
```
    s = SOM(neurons=(5,5), dimentions=3, n_iter=500, learning_rate=0.1)
    s.train(samples)
```

Here 5,5 is the dimention of neurons, 3 is the number of features. samples is numpy array with each sample a 3 dimentional vector