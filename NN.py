# Nicholas Parise, 7242530
import sys
import numpy as np
import csv
import time

weights_hidden = None
bias_hidden = None
weights_output = None
bias_output = None

momentum_wh = None
momentum_bh = None
momentum_wo = None
momentum_bo = None

weights = []
biases = []
momentums = []

results = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def derivative_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

def tanH(x):
    return np.tanh(x)
    # the numpy tanH has some sort of infinity correction, so that one is used
    #return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative_tanH(x):
    return 1 - (tanH(x) ** 2)


def mean_squared(expected, actual):
    return np.mean((expected - actual) ** 2)


def softmax(x):
    ex = np.exp(x - np.max(x, axis=1, keepdims=True))
    return ex / np.sum(ex, axis=1, keepdims=True)


# get classificaiton of output compareed to computed output
# uses the highest value as the selected value.
def classification_rate(predicted, actual):
    pred_classes = np.argmax(predicted, axis=1)
    actual_classes = np.argmax(actual, axis=1)

    correct = np.sum(pred_classes == actual_classes)

    return correct / len(actual_classes)



# read in file and setup input and output array
# This function also normalizes the inputs to be between 0 -> 1
def readFile(file):
    lines = file.read().strip().splitlines()
    rows, dataPoints = map(int, lines[0].split())

    temp_inputs = []
    temp_outputs = []

    for line in lines[1:rows + 1]:
        values = list(map(float, line.split()))
        output_value = values[0]
        input_values = values[1:]

        if output_value == 1:
            temp_outputs.append([1, 0])
        else:
            temp_outputs.append([0, 1])
        
        temp_inputs.append(input_values)

    inputs = np.array(temp_inputs)
    outputs = np.array(temp_outputs)

    # normalize
    min = np.min(inputs, axis=0)
    max = np.max(inputs, axis=0)
    denominator = np.where(max - min == 0, 1, max - min)
    inputs = (inputs - min) / denominator

    if inputs.shape != (rows, dataPoints):
        print(f"Error: Expected ({rows}, {dataPoints}) but got {inputs.shape}")

    return inputs, outputs

# initialize the NN 
def init(input_size, hidden_size, output_size, layers = 5):
    global weights_hidden, bias_hidden, weights_output, bias_output
    # momentum
    global momentum_wh, momentum_bh, momentum_wo, momentum_bo
    
    # 5 layer network
    global weights, biases, momentums
    weights = []
    biases = []
    momentums = []
    
    # allows for differnet sized of layers
    layer_sizes = [input_size] + [hidden_size] * (layers-2) + [output_size]
    #print(layer_sizes)
    for layer_index in range(len(layer_sizes) - 1):
        temp_weight = np.random.uniform(-0.25, 0.25,(layer_sizes[layer_index], layer_sizes[layer_index + 1]))
        temp_bias = np.zeros((1, layer_sizes[layer_index + 1]))

        weights.append(temp_weight)
        biases.append(temp_bias)
        momentums.append(np.zeros_like(temp_weight))

    weights_hidden = np.random.uniform(-0.25, 0.25, (input_size, hidden_size))
    bias_hidden = np.zeros((1, hidden_size))
    weights_output = np.random.uniform(-0.25, 0.25, (hidden_size, output_size))
    bias_output = np.zeros((1, output_size))

    momentum_wh = np.zeros_like(weights_hidden)
    momentum_bh = np.zeros_like(bias_hidden)
    momentum_wo = np.zeros_like(weights_output)
    momentum_bo = np.zeros_like(bias_output)

# reset the NN values
def reset_NN():
    global weights_hidden, bias_hidden, weights_output, bias_output
    # momentum
    global momentum_wh, momentum_bh, momentum_wo, momentum_bo
    # layers
    global weights,biases,momentums

    input_size = weights_hidden.shape[0]
    hidden_size = weights_hidden.shape[1]
    output_size = weights_output.shape[1]

    layers = len(weights)

    weights = []
    biases = []
    momentums = []

    layer_sizes = [input_size] + [hidden_size] * (layers-2) + [output_size]
    for layer_index in range(len(layer_sizes) - 1):
        temp_weight = np.random.uniform(-0.25, 0.25,(layer_sizes[layer_index], layer_sizes[layer_index + 1]))
        temp_bias = np.zeros((1, layer_sizes[layer_index + 1]))

        weights.append(temp_weight)
        biases.append(temp_bias)
        momentums.append(np.zeros_like(temp_weight))


    weights_hidden = np.random.uniform(-0.25, 0.25, (input_size, hidden_size))
    bias_hidden = np.zeros((1, hidden_size))
    weights_output = np.random.uniform(-0.25, 0.25, (hidden_size, output_size))
    bias_output = np.zeros((1, output_size))

    momentum_wh = np.zeros_like(weights_hidden)
    momentum_bh = np.zeros_like(bias_hidden)
    momentum_wo = np.zeros_like(weights_output)
    momentum_bo = np.zeros_like(bias_output)

# Save results into a csv, either the global loss or the training and test set
def save_results(filename_base, k, epochs, show_global = False):
    global results
    global weights_hidden

    hidden_size = weights_hidden.shape[1]

    timestamp = int(time.time())
    filename = f"{filename_base}-h{hidden_size}-{timestamp}.csv"

    # Split results into folds
    fold_size = epochs
    folds = [results[i * fold_size:(i + 1) * fold_size] for i in range(k)]

    # Prepare headers
    headers = []
    for i in range(k):
        if(show_global):
            headers.extend([
                f"Epoch (Fold {i+1})",
                f"Global_Loss (Fold {i+1})",
            ])
        else:
            headers.extend([
                f"Epoch (Fold {i+1})",
                f"Loss_Train (Fold {i+1})",
                f"Loss_Test (Fold {i+1})"
            ])

    # Write CSV
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(fold_size):
            row = []
            for fold in folds:
                if i < len(fold):
                    epoch, train, test = fold[i]
                    global_error = (train + test) / 2
                    if(show_global):
                        row.extend([epoch, f"{global_error:.5f}"])
                    else:
                        row.extend([epoch, f"{train:.5f}", f"{test:.5f}"])

                else:
                    if(show_global):
                        row.extend(["", "", ""])
                    else:
                        row.extend(["", ""])
            writer.writerow(row)

    print(f"saved to {filename}")

# save the classifications to a csv
def save_classification(filename, fold_number, hidden_size, best_test_acc):
    row = [filename, fold_number, hidden_size, f"{best_test_acc:.5f}"]

    # append mode (no header each time)
    with open("classification_results.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# forward propagation for basic 3 layer network
def forward_propagation(inputs):
    global weights_hidden, bias_hidden, weights_output, bias_output

    # multipy input by weights, add bias and pass into activation function
    hidden_in = np.dot(inputs, weights_hidden) + bias_hidden
    hidden_out = sigmoid(hidden_in)
    # multipy output from previus function by weights, add bias and pass into activation function
    output_in = np.dot(hidden_out, weights_output) + bias_output
    output_out = sigmoid(output_in)

    return output_out, hidden_out

# back propagation for basic 3 layer network
def back_propagation(inputs, output, output_out, hidden_out, learning_rate):
    global weights_hidden, bias_hidden, weights_output, bias_output

    #calculate error at output
    output_error = output - output_out
    output_delta = output_error * derivative_sigmoid(output_out)

    #calculate error at hidden layer
    hidden_error = np.dot(output_delta, weights_output.T)
    hidden_delta = hidden_error * derivative_sigmoid(hidden_out)

    # update matrix using error rates
    # wj = wj * a * Err * xj
    weights_output += learning_rate * np.dot(hidden_out.T, output_delta)
    bias_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    
    weights_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
    bias_hidden += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)


# back propagation for basic 3 layer network using momentum 
def back_propagation_momentum(inputs, output, output_out, hidden_out, learning_rate = 0.1, momentum = 0.5):
    global weights_hidden, bias_hidden, weights_output, bias_output
    global momentum_wh, momentum_bh, momentum_wo, momentum_bo

    #calculate error at output
    output_error = output - output_out
    output_delta = output_error * derivative_sigmoid(output_out)

    #calculate error at hidden layer
    hidden_error = np.dot(output_delta, weights_output.T)
    hidden_delta = hidden_error * derivative_sigmoid(hidden_out)

    #calculate momemtum values
    new_momentum_wo = np.dot(hidden_out.T, output_delta)
    new_momentum_bo = np.sum(output_delta, axis=0, keepdims=True)
    new_momentum_wh = np.dot(inputs.T, hidden_delta)
    new_momentum_bh = np.sum(hidden_delta, axis=0, keepdims=True)

    # update matrix using error rates
    weights_output += learning_rate * new_momentum_wo + momentum * momentum_wo
    bias_output += learning_rate * new_momentum_bo + momentum * momentum_bo
    
    weights_hidden += learning_rate * new_momentum_wh + momentum * momentum_wh
    bias_hidden += learning_rate * new_momentum_bh + momentum * momentum_bh

    # save update values
    momentum_wh = learning_rate * new_momentum_wh + momentum * momentum_wh
    momentum_bh = learning_rate * new_momentum_bh + momentum * momentum_bh
    momentum_wo = learning_rate * new_momentum_wo + momentum * momentum_wo
    momentum_bo = learning_rate * new_momentum_bo + momentum * momentum_bo

# forward propagation for an n sized neural network
def forward_propagation_n(inputs):
    global weights,biases,momentums

    current_activation = inputs
    all_activations = [current_activation]
    all_weighted_inputs = []

    for layer in range(len(weights)):
        weighted_in =  np.dot(current_activation, weights[layer]) + biases[layer]
        current_activation = tanH(weighted_in)

        all_weighted_inputs.append(weighted_in)
        all_activations.append(current_activation)
    return all_activations, all_weighted_inputs


# back propagation for an n sized neural network (with momentum)
def back_propagation_n(output, all_activations, all_weighted_inputs, learning_rate = 0.1, momentum_rate = 0.5):
    global weights,biases,momentums

    layers = len(weights)
    all_deltas = [None] * layers

    # output layer error delta
    output_error = output - all_activations[-1]
    all_deltas[-1] = output_error * derivative_tanH(all_activations[-1])

    # hidden layer error deltas
    for layer in range(layers - 2, -1, -1):
        layer_delta = all_deltas[layer+1]
        layer_weights = weights[layer+1]

        error = np.dot(layer_delta, layer_weights.T)
        all_deltas[layer] = error * derivative_tanH(all_activations[layer+1])

    # weight updates
    for layer in range(layers):
        
        current_grad = np.dot(all_activations[layer].T, all_deltas[layer])
        momentums[layer] = learning_rate * current_grad + momentum_rate * momentums[layer]

        weights[layer] += momentums[layer]
        biases[layer]  += learning_rate * np.sum(all_deltas[layer], axis=0, keepdims=True)

# train 3 layer neural network to completion
def train(inputs, outputs, epochs=1000, learning_rate=0.1):
    DECAY = 0.0001
    for epoch in range(epochs):
        output_out, hidden_out = forward_propagation(inputs)
        dynamic_lr = learning_rate / (1.0 + (DECAY * epoch))
        back_propagation(inputs, outputs, output_out, hidden_out, dynamic_lr)

        if epoch % 50 == 0:
            loss = mean_squared(outputs, output_out)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# train 3 layer neural network, train with train sets and comparing with test sets 
# also compute dynamic learning rate and dynamic momentum 
def train_testing(inputs, outputs, test_inputs, test_outputs ,epochs=1000, learning_rate=0.1, base_momentum = 0.5):
    global results
    best_test_class = 0.0
    DECAY = 0.0001
    max_error = 0.0
    print(f"Epoch, loss_training, loss_testing")
    for epoch in range(epochs):
        output_out, hidden_out = forward_propagation(inputs)
        testing_output_out = forward_propagation(test_inputs)[0]
        
        train_acc = classification_rate(output_out, outputs)
        test_acc = classification_rate(testing_output_out, test_outputs)

        if test_acc > best_test_class:
            best_test_class = test_acc

        dynamic_lr = learning_rate / (1.0 + (DECAY * epoch))
        #dynamic_lr = learning_rate * np.exp(-0.001 * epoch)

        loss_training = mean_squared(outputs, output_out)        
        loss_testing = mean_squared(test_outputs, testing_output_out)
        global_error = (loss_training + loss_testing) / 2

        # dynamic momentum
        #if epoch == 0:
        if global_error > max_error:
            max_error = global_error

        momentum = base_momentum * (global_error / max_error)
        momentum = max(0, min(base_momentum, momentum))

        # Part B Disable momentum
        if(base_momentum == 0):
            momentum = 0

        # Part C momentum
        back_propagation_momentum(inputs, outputs, output_out, hidden_out, dynamic_lr,momentum)

        results.append((epoch, loss_training, loss_testing))
        #print(f"{epoch},{loss_training:.4f},{loss_testing:.4f}")
        #print(dynamic_lr)
    return best_test_class



# train n layer neural network, train with train sets and comparing with test sets 
# also compute dynamic learning rate and dynamic momentum 
def train_testing_n(inputs, outputs, test_inputs, test_outputs ,epochs=1000, learning_rate=0.1, base_momentum = 0.5):
    global results
    best_test_class = 0.0
    DECAY = 0.0001
    max_error = 1.0
    print(f"Epoch, loss_training, loss_testing")
    for epoch in range(epochs):
        all_activations, all_weighted_inputs = forward_propagation_n(inputs)
        testing_all_activations = forward_propagation_n(test_inputs)[0]
        
        # Part E Apply softmax 
        testing_all_activations[-1] = softmax(testing_all_activations[-1])
        all_activations[-1] = softmax(all_activations[-1])

        train_acc = classification_rate(all_activations[-1], outputs)
        test_acc = classification_rate(testing_all_activations[-1], test_outputs)

        if test_acc > best_test_class:
            best_test_class = test_acc

        #dynamic_lr = learning_rate / (1.0 + (DECAY * epoch))
        dynamic_lr = learning_rate * np.exp(-0.0005 * epoch)

        loss_training = mean_squared(outputs, all_activations[-1])        
        loss_testing = mean_squared(test_outputs, testing_all_activations[-1])
        global_error = (loss_training + loss_testing) / 2

        # dynamic momentum
        if epoch == 0:
        #if global_error > max_error:
            max_error = global_error

        momentum = base_momentum * (global_error / max_error)
        momentum = max(0, min(base_momentum, momentum))

        # Part B Disable momentum
        if(base_momentum == 0):
            momentum = 0

        # Part C momentum
        back_propagation_n(outputs, all_activations, all_weighted_inputs, dynamic_lr, momentum)

        results.append((epoch, loss_training, loss_testing))
    return best_test_class

# create k folds with input,

def k_fold(inputs, outputs, k=3, epochs=1000, learning_rate=0.1, base_momentum = 0.5, filename = "null"):
    
    global weights_hidden
    hidden_size = weights_hidden.shape[1]

    global results
    results.clear()

    # get top 4 similar inputs to keep on training set
    # when inputs are similar and outputs are opposite
    n = len(inputs)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(inputs[i] - inputs[j])
            if outputs[i, 0] == 1 and outputs[j, 1] == 1 or outputs[i, 1] == 1 and outputs[j, 0] == 1:
                pairs.append((dist, i, j))

    pairs.sort(key=lambda x: x[0])
    # get index
    close_idx = set()
    for dist, i, j in pairs[:2]:
        close_idx.add(i)
        close_idx.add(j)
    close_idx = np.array(list(close_idx))
    
    all_good_idx = np.where(outputs[:, 0] == 1)[0]
    all_bad_idx = np.where(outputs[:, 1] == 1)[0]
    
    good_idx = np.array([i for i in all_good_idx if i not in close_idx])
    bad_idx  = np.array([i for i in all_bad_idx  if i not in close_idx])
    
    np.random.shuffle(good_idx)
    np.random.shuffle(bad_idx)

    good_splits = np.array_split(good_idx, k)
    bad_splits = np.array_split(bad_idx, k)    

    folds = []
    for i in range(k):
        fold_idx = np.concatenate((good_splits[i], bad_splits[i]))
        np.random.shuffle(fold_idx)

        foldX = inputs[fold_idx] 
        foldY = outputs[fold_idx]
        folds.append((foldX, foldY))

    total_loss = 0.0
    for i in range(k):
        print(f"\n--- Fold {i+1}/{k} ---")
        # make index the test fold, and concatenate all other
        testing_input =  folds[i][0] 
        testing_output = folds[i][1]
        training_input = np.concatenate([folds[j][0] for j in range(k) if j != i] + [inputs[close_idx]])
        training_output = np.concatenate([folds[j][1] for j in range(k) if j != i] + [outputs[close_idx]])
        reset_NN()
        # Part C and Part D
        #best_test = train_testing(training_input, training_output, testing_input, testing_output, epochs, learning_rate, base_momentum) 
        # Part E, Part F
        best_test = train_testing_n(training_input, training_output, testing_input, testing_output, epochs, learning_rate, base_momentum)

        # Part D, Part E, Part F
        save_classification(filename, i+1, hidden_size, best_test)



if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=6)
    
    filename = sys.argv[1]
    print(filename)
    file = open(filename)

    inputs, outputs = readFile(file)

    # set NN size appropriately for
    input_size = inputs.shape[1]
    hidden_size = int((input_size)/4)
    OUTPUTS = 2
    
    #part E & F
    
    for hidden_sizes in [5,10,15,20,25,30]:
        np.random.seed(int(time.time()) + hidden_sizes)
        print(f"Hidden size: {hidden_sizes}")
        init(input_size, hidden_sizes, OUTPUTS, layers=5)
        k_fold(inputs,outputs,3,1000,0.01, 0.5, filename)
        save_results(filename, 3, 1000, show_global=False)
    

    #part D
    '''
    for hidden_sizes in [int((input_size)/6),int((input_size)/4), int((input_size)/2), int((3*input_size)/4)]:
        np.random.seed(int(time.time()) + hidden_sizes)
        print(f"Hidden size: {hidden_sizes}")
        init(input_size, hidden_sizes, OUTPUTS)
        k_fold(inputs,outputs,3,1000,0.1, 0.9, filename)
    '''

    #part C
    '''
    for hidden_sizes in [int((input_size)/10),int((input_size)/8),int((input_size)/6),int((input_size)/4), int((input_size)/2), int((3*input_size)/4)]:
        np.random.seed(int(time.time()) + hidden_sizes)
        print(f"Hidden size: {hidden_sizes}")
        init(input_size, hidden_sizes, OUTPUTS)
        k_fold(inputs,outputs,3,1000,0.1, 0.5)
        save_results(filename, 3, 1000, show_global=False)
    '''
    # Part B
    '''
    for hidden_sizes in [int((input_size)/10),int((input_size)/8),int((input_size)/6),int((input_size)/4), int((input_size)/2), int((3*input_size)/4)]:
        np.random.seed(int(time.time()) + hidden_sizes)
        print(f"Hidden size: {hidden_sizes}")
        init(input_size, hidden_sizes, OUTPUTS)
        k_fold(inputs,outputs,3,1000,0.1, 0.0)
        save_results(filename, 3, 1000, show_global=False)
    '''
    # Part A
    '''
    init(input_size, hidden_size, OUTPUTS)
    train(inputs, outputs, 1000, 0.1)
    '''