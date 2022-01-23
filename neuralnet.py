import numpy as np
import sys


class NeuralNetwork:
    def __init__(self, input_nodes, output_nodes, hidden_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes
        self.lr = learning_rate

        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                                      (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                                        (self.hidden_nodes, self.output_nodes))

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))  # sigmoid
        self.loss_function = lambda y, Y: np.mean((y - Y) ** 2)  # mean-squared error

  
    def train(self, features, targets, epochs, val_features, val_targets, batch_size=100):

        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        self.losses = {'train':[], 'validation':[]}

        for epoch in range(epochs):
            batch = np.random.choice(range(features.shape[0]), size=batch_size)
            batch_features, batch_targets = features[batch], targets[batch]
            
            delta_weights_i_h *= 0
            delta_weights_h_o *= 0

            for X, y in zip(batch_features, batch_targets):
                hidden_outputs, final_outputs = self.forward(X)

                delta_weights_i_h, delta_weights_h_o = self.backpropagation(X, y, final_outputs, hidden_outputs,
                                                                            delta_weights_i_h, delta_weights_h_o)
            
            self.update_weights(delta_weights_i_h, delta_weights_h_o, batch_size)

            output_train = self.predict(features).T
            output_val = self.predict(val_features).T

            train_loss = self.loss_function(output_train, targets)
            val_loss = self.loss_function(output_val, val_targets)

            sys.stdout.write("\rEpoch {}/{}".format(epoch + 1, epochs) \
                            + "\tTraining Loss:" + str(train_loss)[:5]
                            + "\tValidation Loss:" + str(val_loss)[:5])

            sys.stdout.flush()

            self.losses['train'].append(train_loss)
            self.losses['validation'].append(val_loss)


    def forward(self, X):

        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_input = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_input # continuous

        return hidden_outputs, final_outputs


    def backpropagation(self, X, y, final_output, hidden_output, delta_weights_i_h, delta_weights_h_o):
        
        error = y - final_output
        output_error_term = error # continuous

        hidden_error = np.dot(error, self.weights_hidden_to_output.T)
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        delta_weights_i_h += hidden_error_term * X[:, None]
        delta_weights_h_o += output_error_term * hidden_output[:, None]

        return delta_weights_i_h, delta_weights_h_o

    
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

        return

    
    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_to_hidden)
        hidden_output = self.activation_function(hidden_input)

        final_input = np.dot(hidden_output, self.weights_hidden_to_output)
        final_output = final_input # continuous

        return final_output