import numpy as np
import nnfs 
from nnfs.datasets import spiral_data 

nnfs.init()

X , y = spiral_data(100,3)

'''

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
'''

'''
inputs = [0, 2 , -1 , 3.3 , -2.7 , 1.1 , 2.2, -100 ]

output = []

for i in inputs :
    output.append(max(0, i))

print(output)
'''

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))  # Corrected here
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)



layer1 = Layer_Dense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

