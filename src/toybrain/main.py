import numpy as np
from csv import reader

np.random.seed(1)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
     network = list()
     # Hidden layer
     # If n_inputs = 2 and n_hidden = 1 then [{'weights': [0.417022004702574, 0.7203244934421581, 0.00011437481734488664]}]
     hidden_layer = [{'weights':[np.random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
     network.append(hidden_layer)

     # Output layer
     # If n_hidden = 1 and n_outputs = 2 then [{'weights': [0.30233257263183977, 0.14675589081711304]}, {'weights': [0.0923385947687978, 0.1862602113776709]}]
     output_layer = [{'weights':[np.random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
     network.append(output_layer)
     return network

# --- Forward Propagate

# Calculate neuron activation for an input
def activate(weights, inputs):
     activation = weights[-1]
     for i in range(len(weights)-1):
          w = weights[i] * inputs[i]
          # print(f"w: {w}")
          activation += w
     return activation

# Transfer neuron activation
def transfer(activation):
     # sigmoid activation function
     return 1.0 / (1.0 + np.exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
     inputs = row
     for layer in network:
          new_inputs = []
          for neuron in layer:
               activation = activate(neuron['weights'], inputs)
               neuron['output'] = transfer(activation)
               new_inputs.append(neuron['output'])
          inputs = new_inputs
     return inputs

# --- Back Propagate Error

# Calculate the derivative of an neuron output
def transfer_derivative(output):
     return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
     for i in reversed(range(len(network))):
          layer = network[i]
          errors = list()
          if i != len(network)-1:
               for j in range(len(layer)):
                    error = 0.0
                    for neuron in network[i + 1]:
                         error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
          else:
               for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(neuron['output'] - expected[j])
          for j in range(len(layer)):
               neuron = layer[j]
               neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# --- Train Network

# Update network weights with error
def update_weights(network, row, l_rate):
     for i in range(len(network)):
          inputs = row[:-1]
          if i != 0:
               inputs = [neuron['output'] for neuron in network[i - 1]]
          for neuron in network[i]:
               for j in range(len(inputs)):
                    neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
               neuron['weights'][-1] -= l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
     for epoch in range(n_epoch):
          sum_error = 0
          for row in train:
               outputs = forward_propagate(network, row)
               expected = [0 for i in range(n_outputs)]
               expected[row[-1]] = 1
               sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
               backward_propagate_error(network, expected)
               update_weights(network, row, l_rate)
          print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

dataset = [
     # format = [Input, Input, Output]
     [2.7810836,2.550537003,0],
     [1.465489372,2.362125076,0],
     [3.396561688,4.400293529,0],
     [1.38807019,1.850220317,0],
     [3.06407232,3.005305973,0],
     [7.627531214,2.759262235,1],
     [5.332441248,2.088626775,1],
     [6.922596716,1.77106367,1],
     [8.675418651,-0.242068655,1],
     [7.673756466,3.508563011,1],
]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)

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
for layer in network:
     print(layer)
train_network(network, dataset, 0.5, 200, n_outputs)
# After training
print("After training")
for layer in network:
     print(layer)


# --- Predict

# Make a prediction with a network
def predict(network, row):
     outputs = forward_propagate(network, row)
     return outputs.index(max(outputs))

for row in dataset:
     prediction = predict(network, row)
     print('Expected=%d, Got=%d' % (row[-1], prediction))


# --- CSV Dataset utils
# Load a CSV file
def load_csv(filename):
     dataset = list()
     with open(filename, 'r') as file:
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

# Find the min and max values for each column
def dataset_minmax(dataset):
     minmax = list()
     stats = [[min(column), max(column)] for column in zip(*dataset)]
     return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
     for row in dataset:
          for i in range(len(row)-1):
               row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
     dataset_split = list()
     dataset_copy = list(dataset)
     fold_size = int(len(dataset) / n_folds)
     for i in range(n_folds):
          fold = list()
          while len(fold) < fold_size:
               index = np.random.randint(len(dataset_copy))
               fold.append(dataset_copy.pop(index))
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
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
     folds = cross_validation_split(dataset, n_folds)
     scores = list()
     for fold in folds:
          train_set = list(folds)
          train_set.remove(fold)
          train_set = sum(train_set, [])
          test_set = list()
          for row in fold:
               row_copy = list(row)
               test_set.append(row_copy)
               row_copy[-1] = None
          predicted = algorithm(train_set, test_set, *args)
          actual = [row[-1] for row in fold]
          accuracy = accuracy_metric(actual, predicted)
          scores.append(accuracy)
     return scores

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
     n_inputs = len(train[0]) - 1
     n_outputs = len(set([row[-1] for row in train]))
     network = initialize_network(n_inputs, n_hidden, n_outputs)
     train_network(network, train, l_rate, n_epoch, n_outputs)
     predictions = list()
     for row in test:
          prediction = predict(network, row)
          predictions.append(prediction)
     return(predictions)


filename = 'dataset/seeds_dataset.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
     str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = 5
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
