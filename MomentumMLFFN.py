import numpy as np
import pickle
from Neuron import MomentumNeuron, derivative
from tqdm import tqdm
import matplotlib.pyplot as plt

class MLFFNN():
    def __init__(self, input_size=None, hidden_layers=None, hidden_sizes=None, output_size=None, input_data=None,
                 labels=None, num_epochs=None, learning_rate=None, momentum_rate=None, loss="MSE", features_test=None, labels_test=None,
                 idx=None, seed=7246325, savedNetwork = None):
        np.random.seed(seed)
        if savedNetwork:
            self.importNetwork(savedNetwork)
        else:
            self.W = [np.random.uniform(low=-0.1, high=0.1, size=(input_size, hidden_sizes[0]))]
            for i in range(1, hidden_layers):
                self.W.append(np.random.uniform(low=-0.1, high=0.1, size=(hidden_sizes[i-1], hidden_sizes[i])))
            self.W.append(np.random.uniform(low=-0.1, high=0.1, size=(hidden_sizes[-1], output_size)))
            self.hidden_layers = [[MomentumNeuron(momentum_rate) for _ in range(hidden_sizes[i])] for i in range(hidden_layers)]
            self.output_neurons = [MomentumNeuron(momentum_rate) for _ in range(output_size)]
            for layer in self.hidden_layers:
                for neuron in layer:
                    neuron.bias = np.random.uniform(low=-0.1, high=0.1)
            for neuron in self.output_neurons:
                neuron.bias = np.random.uniform(low=-0.1, high=0.1)
            self.input_data = input_data
            self.labels = labels
            self.num_epochs = num_epochs
            self.learning_rate = learning_rate
            self.trainingProgress = []
            self.testPerformance = []
            self.testEvals = []
            self.loss = loss
            self.features_test = features_test
            self.labels_test = labels_test
            self.idx = idx
            self.momentum_rate = momentum_rate


    def softmax(self, z):
        z_shifted = z - np.max(z)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=0)

    def forward_pass(self, input_data):
        hidden_outputs = [input_data]  # Store input as the first "output"
        for i, layer in enumerate(self.hidden_layers):
            current_input = hidden_outputs[-1]  # Use the last output as the current input
            hidden_output = np.array(
                [neuron.activate(current_input, self.W[i][:, j]) for j, neuron in enumerate(layer)])
            hidden_outputs.append(hidden_output)  # Store output of this layer
        raw_output = np.array(
            [neuron.activate(hidden_outputs[-1], self.W[-1][:, i]) for i, neuron in enumerate(self.output_neurons)])
        return raw_output, hidden_outputs[1:]  # Return outputs from hidden layers (exclude input)

    def compute_loss(self, y_true, y_pred):
        #MSE
        if self.loss == "MSE":
            return 0.5 * np.sum((y_true - y_pred) ** 2)
        else:
            return -np.sum(y_true * np.log(y_pred))

    def backpropagate(self, input_set, hidden_outputs, y_true, y_pred, learning_rate, prev_deltaW):
        # Compute gradients for output layer
        dL_dz_output = y_pred - y_true
        dL_dW = [np.outer(hidden_outputs[-1], dL_dz_output)]

        # Initialize dL_dz for the output layer
        dL_dz = dL_dz_output

        new_deltaW = []

        # Compute gradients for hidden layers (iterate in reverse)
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            hidden_layer_output = hidden_outputs[i]
            dL_dz = np.dot(self.W[i + 1], dL_dz) * np.array([derivative(h) for h in hidden_layer_output])

            previous_output = input_set if i == 0 else hidden_outputs[i - 1]
            dL_dW.insert(0, np.outer(previous_output, dL_dz))

            # Update biases for the current hidden layer
            for j, neuron in enumerate(self.hidden_layers[i]):
                delta_b = learning_rate * dL_dz[j]
                neuron.update_bias(delta_b)

        # Update weights
        for i in range(len(self.W)):
            delta_W = learning_rate * dL_dW[i]
            total_delta_W = delta_W + self.momentum_rate * prev_deltaW[i]
            self.W[i] -= total_delta_W
            new_deltaW.append(total_delta_W)

        # Update biases for the output neurons
        for i, neuron in enumerate(self.output_neurons):
            delta_b = learning_rate * dL_dz_output[i]
            neuron.update_bias(delta_b)

        return new_deltaW

    #Questions for Bockus:
    #Backprop after each input or after all inputs?
    #When you say softmax at the output layer, do you literally mean softmax the final output? Or is there more to it?
    #what do you mean by avg global loss?
    def train(self, networkNumber):
        features_test = self.features_test
        labels_test = self.labels_test
        labels_one_hot = np.eye(len(self.output_neurons))[self.labels]
        epoch_progress = tqdm(range(self.num_epochs), desc="Training Progress", unit="epoch")

        # Initialize previous deltaW
        prev_deltaW = [np.zeros_like(w) for w in self.W]

        for epoch in epoch_progress:
            epoch_loss = 0
            test_loss = 0
            p = np.random.permutation(len(self.input_data))
            self.input_data = self.input_data[p]
            self.labels = self.labels[p]
            labels_one_hot = labels_one_hot[p]
            for i in range(len(self.input_data)):
                input_set = self.input_data[i]
                y_true = labels_one_hot[i]
                # Forward pass
                y_pred, hidden_outputs = self.forward_pass(input_set)

                # Compute loss
                loss = self.compute_loss(y_true, y_pred)
                epoch_loss += loss

                # Don't backprop on last epoch
                if epoch == self.num_epochs - 1:
                    continue
                prev_deltaW = self.backpropagate(input_set, hidden_outputs, y_true, y_pred, self.learning_rate,
                                                 prev_deltaW)

            test_correct = 0
            test_incorrect = 0
            labels_test_one_hot = np.eye(len(self.output_neurons))[labels_test]
            for i in range(len(labels_test)):
                test_true = labels_test_one_hot[i]
                test_input = features_test[i]
                test_pred, _ = self.forward_pass(test_input)
                test_loss += self.compute_loss(test_true, test_pred)
                if np.argmax(test_pred) == labels_test[i]:
                    test_correct += 1
                else:
                    test_incorrect += 1
            self.testEvals.append(test_incorrect)
            self.testPerformance.append(test_loss / len(self.features_test))
            self.trainingProgress.append(epoch_loss / len(self.input_data))
            epoch_progress.set_postfix(loss=epoch_loss / len(self.input_data))

        self.exportNetwork(f"network{str(networkNumber)}.pkl")
        self.graphTraining(f"loss{str(networkNumber)}.png")
        self.graphTest(f"testLoss{str(networkNumber)}.png")
        return self.W

    def exportNetwork(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def importNetwork(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)


    def graphTest(self, name):
        plt.clf()
        fig, axs = plt.subplots(2)
        #plot test loss
        axs[0].plot(self.testPerformance)
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epoch')
        #plot test accuracy
        axs[1].plot(self.testEvals)
        axs[1].set_ylabel('Accuracy')
        axs[1].set_xlabel('Epoch')
        plt.savefig(name)
        plt.close()
        return

    def graphTraining(self, name):
        plt.clf()
        plt.plot(self.trainingProgress)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(name)
        plt.close()
        return
