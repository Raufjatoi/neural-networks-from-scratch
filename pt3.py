import numpy as np 


'''
inputs = [1,2,3,2.5]
weights = [0.2,0.8,-0.5,1.0]
bias = 2
'''
inputs = [1,2,3,2.5]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2, 3, 0.5]


output = np.dot(weights,inputs) + biases
print(output)



'''
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2, 3, 0.5]
'''







## THE STUFFS TO UNDERSTAND ##

#bias1 = 2
#bias2 = 3
#bias3 = 0.5




#for example 
'''
some_value = 0.5
weight = -0.7
bias = 0.7

print(some_value*weight)
print(some_value+bias)

'''


'''
layer_outputs = []  # output of current layer

# Iterate over each neuron's weights and bias
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # output of given neuron
    
    # Calculate the weighted sum of inputs
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    
    # Add the neuron's bias
    neuron_output += neuron_bias
    
    # Append the neuron's output to the layer's output list
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''





