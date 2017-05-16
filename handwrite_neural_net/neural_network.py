import numpy as np
import scipy.special

class NeuralNetwork(object):
    def __init__(self, input_n, hidden_n, output_n, learning_rate):
        self.inodes = input_n
        self.hnodes = hidden_n
        self.onodes = output_n
        self.lr = learning_rate
        # Matrices with sophisticated weights
        # Weights from a normal probability distribution
        # Distribution around zero with standard deviation 1/sqrt(incoming links)
        self.wih = np.random.normal(0.0, self.hnodes**(-.05), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, self.onodes**(-.05), (self.onodes, self.hnodes))
        # Activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

    def feed(self, inputs, hidden_outputs=[]):
        # Calculate hidden layer
        if not len(hidden_outputs):
            hidden_inputs = np.dot(self.wih, inputs)
            hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate last layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_outputs = self.activation_function(np.dot(self.wih, inputs))
        final_outputs = self.feed(inputs, hidden_outputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        change_wh = output_errors * final_outputs * (1.0 - final_outputs)
        change_wh = np.dot(change_wh, np.transpose(hidden_outputs))
        self.who += self.lr * change_wh

        change_wi = hidden_errors * hidden_outputs * (1.0 - hidden_outputs)
        change_wi = np.dot(change_wi, np.transpose(inputs))
        self.wih += self.lr * change_wi

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        outputs = self.feed(inputs)
        return outputs
