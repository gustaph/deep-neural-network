import numpy as np


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