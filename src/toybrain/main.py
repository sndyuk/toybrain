import numpy as np
from nptyping import NDArray, Int, Float, Shape
from csv import reader

np.random.seed(1)
np.set_printoptions(precision=4, suppress=True)


class Neuron:
    def __init__(self, weights: NDArray[Shape["1"], Float]):
        self.weights = weights
        self.output = 0.0
        self.delta = 0.0
    
    def __repr__(self):
        return f"Neuron(weights={self.weights}, output={self.output}, delta={self.delta})"
    
    def __str__(self):
        return self.__repr__()


class Network:
    def __init__(self, hidden_layer: list[Neuron], output_layer: list[Neuron]):
        self.layers = [hidden_layer, output_layer]
    
    def __repr__(self):
        return f"Network({self.layers})"
    
    def __str__(self):
        return self.__repr__()


# Initialize a network
def initialize_network(n_inputs: int, n_hidden: int, n_outputs: int):
    hidden_layer = [Neuron(np.random.rand(n_inputs + 1)) for _ in range(n_hidden)]
    output_layer = [Neuron(np.random.rand(n_hidden + 1)) for _ in range(n_outputs)]
    return Network(hidden_layer, output_layer)


# --- Forward Propagate


# Calculate neuron activation for an input
def activate(weights: NDArray[Shape["1"], Float], inputs: NDArray[Shape["1"], Float]) -> float:
    return weights[-1] + np.dot(weights[:-1], inputs[:weights.shape[0] - 1])


# Transfer neuron activation
def transfer(activation: float) -> float:
    # sigmoid activation function
    return 1.0 / (1.0 + np.exp(-activation))


# Forward propagate input to a network output
def forward_propagate(network: Network, row: NDArray[Shape["1"], Float]) -> NDArray[Shape["1"], Float]:
    inputs = row
    for layer in network.layers:
        new_inputs = np.zeros(len(layer))
        for i, neuron in enumerate(layer):
            activation = activate(neuron.weights, inputs)
            neuron.output = transfer(activation)
            new_inputs[i] = neuron.output
        inputs = new_inputs
    return inputs


# --- Back Propagate Error


# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)


# Backpropagate error and store in neurons
def backward_propagate_error(network: Network, expected: NDArray[Shape["1"], Int]):
    for i in reversed(range(len(network.layers))):
        layer = network.layers[i]
        if i != len(network.layers) - 1:
            # Hidden layers
            errors = [
                sum(neuron.weights[j] * neuron.delta for neuron in network.layers[i + 1])
                for j, _ in enumerate(layer)
            ]
        else:
            # Output layer
            errors = [
                neuron.output - v
                for neuron, v in zip(layer, expected)
            ]
        for j, neuron in enumerate(layer):
            neuron.delta = errors[j] * transfer_derivative(neuron.output)


# --- Train Network


# Update network weights with error
def update_weights(network: Network, row: NDArray[Shape["1"], Float], l_rate: float):
    for i, layer in enumerate(network.layers):
        inputs = [neuron.output for neuron in network.layers[i - 1]] if i != 0 else row[:-1]
        
        for neuron in layer:
            for j, v in enumerate(inputs):
                neuron.weights[j] -= l_rate * neuron.delta * v
            neuron.weights[-1] -= l_rate * neuron.delta


# Train a network for a fixed number of epochs
def train_network(network: Network, train: NDArray, l_rate: float, n_epoch: int, n_outputs: int):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = np.zeros(n_outputs, dtype=int)
            expected[row[-1].astype(int)] = 1
            sum_error += np.sum((expected - outputs) ** 2)
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        if epoch % 50 == 0:
            print(">epoch=%d, error=%.3f" % (epoch, sum_error))


dataset = np.array([
    # format = [Input, Input, Output]
    [2.7810836, 2.550537003, 0],
    [1.465489372, 2.362125076, 0],
    [3.396561688, 4.400293529, 0],
    [1.38807019, 1.850220317, 0],
    [3.06407232, 3.005305973, 0],
    [7.627531214, 2.759262235, 1],
    [5.332441248, 2.088626775, 1],
    [6.922596716, 1.77106367, 1],
    [8.675418651, -0.242068655, 1],
    [7.673756466, 3.508563011, 1],
], dtype=float)

n_inputs = len(dataset[0]) - 1
n_outputs = len(np.unique(dataset[:, -1]))
network = initialize_network(n_inputs, 3, n_outputs)

# test the forward propagation
# row = [0, 0, None]
# output = forward_propagate(network, row)
# print(output)

# test backpropagation of error
# expected = [0, 1]
# backward_propagate_error(network, expected)
# for layer in network:
#      print(layer)


# Before training
print("Before training")
for layer in network.layers:
    print(layer)
train_network(network, dataset, 0.5, 200, n_outputs)
# After training
print("After training")
for layer in network.layers:
    print(layer)


# --- Predict


# Make a prediction with a network
def predict(network: Network, row: NDArray[Shape["1"], Float]):
    outputs = forward_propagate(network, row)
    return np.argmax(outputs)


for row in dataset:
    prediction = predict(network, row)
    print("Expected=%d, Got=%d" % (row[-1], prediction))


# --- CSV Dataset utils
# Load a CSV file
def load_csv(filename):
    dataset = []
    with open(filename, "r") as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset: list[list[float]]):
    # Find the min and max values for each column
    minmax = [(min(column), max(column)) for column in zip(*dataset)]

    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset: list[list[float]], n_folds: int) -> list[list[list[float]]]:
    dataset_split: list[list[list[float]]] = []
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold: list[list[float]] = []
        while len(fold) < fold_size:
            index = np.random.randint(len(dataset))
            fold.append(dataset[index])
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset: list[list[float]], algorithm, n_folds: int, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []

    for i, fold in enumerate(folds):
        # Combine training sets and flatten
        train_set = [row for j, f in enumerate(folds) if j != i for row in f]

        # Prepare test set by replacing the last column value with NaN
        test_set = [row[:-1] + [float('nan')] for row in fold]

        # Convert to numpy arrays and make them read-only
        train_np, test_np = np.array(train_set, dtype=float), np.array(test_set, dtype=float)
        for arr in (train_np, test_np):
            arr.flags.writeable = False

        # Predict and calculate accuracy
        predicted = algorithm(train_np, test_np, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)

        scores.append(accuracy)

    return scores


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train: NDArray, test: NDArray, l_rate: float, n_epoch: int, n_hidden: int):
    n_inputs = len(train[0]) - 1
    n_outputs = len(np.unique(train[:, -1]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = []
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions

filename = "dataset/seeds_dataset.csv"
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
# normalize input variables
normalize_dataset(dataset)

# evaluate algorithm
n_folds = 3
l_rate = 0.3
n_epoch = 700
n_hidden = 5
scores = evaluate_algorithm(
    dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden
)
print("Scores: %s" % scores)
print("Mean Accuracy: %.3f%%" % (sum(scores) / float(len(scores))))
