import numpy as np
from neural_network import NeuralNetwork

epochs = 5

input_nodes = 784
hidden_nodes = 50
output_nodes = 10
learning_rate = 0.1

net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

def train():
    with open('../datasets/mnist_dataset/mnist_train.csv', 'r') as fin:
        for line in fin:
            l = line.split(',')
            image = (np.asfarray(l[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(l[0])] = 0.99
            net.train(image, targets)

def test():
    scores = []
    with open('../datasets/mnist_dataset/mnist_test.csv', 'r') as fin:
        for line in fin:
            l = line.split(',')
            image = (np.asfarray(l[1:]) / 255.0 * 0.99) + 0.01
            output = net.query(image)
            label = np.argmax(output)
            scores.append(int(label == int(l[0])))
    return np.asarray(scores)

def main():
    train()
    print('Epoch ready')
    scores = test()
    print("Performance =", 1.0 * scores.sum() * 100 / scores.size)

if __name__ == '__main__':
    main()
