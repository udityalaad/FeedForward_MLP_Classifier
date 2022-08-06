from cmath import nan
from fnmatch import fnmatchcase
import numpy as np
import pickle
import math
from acc_calc import accuracy
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------------------------------
#       Multi-Layer Perceptron (MLP) - Neural Network
# -------------------------------------------------------------------------------------------------
class MLP_Neural_Nw ():
    def __init__ (self, size_input_features = 0, size_hidden = 0, size_out = 0, n = 0, activations = [None, None], loss_function = None, max_epochs = 0, test_validation_division = 0, individual_acceptable_loss = 0):
        # Number of features in each input - [(+1) - for Bias at Hidden Layer]
        self.size_input_features = size_input_features + 1

        # Number of neurons in the hidden layer
        self.size_hidden = size_hidden

        # Number of neurons in the output layer
        self.size_out = size_out

        # Learning rate
        self.n = n

        # Activation Functions
        self.hidden_activation = activations[0]
        self.out_activation = activations[1]

        # Activation Object
        self.activation = Activation()

        # Loss Object
        self.loss = Loss(loss_function)

        # Division of training data
        self.division = test_validation_division

        # Max Allowed Epochs (while training)
        self.max_epochs = max_epochs


        # Weights
        # Weights from input-to-hidden layer
        self.w_IH = np.random.randn(self.size_hidden, self.size_input_features)     
        # Weights from hidden-to-output layer - [(+1) for Bias at Output Layer]
        self.w_HO = np.random.randn(self.size_out, self.size_hidden + 1)    


        # Output Arrays - for actual computed values
        self.actual_in = None
        self.actual_hidden = None
        self.actual_out = None

        # Arrays - to store Error Signals
        self.es_out = None
        self.es_hidden = None

        # Acceptable Loss:
        self.individual_acceptable_loss = individual_acceptable_loss
        self.cumulative_acceptable_loss = 0.02


    # Starting Point - To Build the Model
    def build_Model (self, input_filename, label_filename):
        # -------------- Divide Training Data into - Test and Validation sets -----------------
        # Load Input Patterns and Output Labels - from file
        input = np.loadtxt(input_filename, delimiter=",", dtype=float)
        label = np.loadtxt(label_filename, delimiter=",", dtype=float)

        # Split the data into test and validation sets
        test_input_set, validation_input_set, test_label_set, validation_label_set = train_test_split(input, label, test_size=(1-self.division), random_state=40)

        # Set acceptable Cumulative_Loss
        self.cumulative_acceptable_loss = len(test_input_set) * self.individual_acceptable_loss
        # ------------------------------------------------------------
        
        # -------------- Create Multi-layer Perceptron (MLP) Network - Model -----------------
        self.train_MLP(test_input_set, test_label_set)
        # ------------------------------------------------------------

        # -------------- Test and Validate the model -----------------
        predictions_test = self.predict(test_input_set)
        print("Accuracy of Test Set: " + str(accuracy(test_label_set, predictions_test) * 100))

        predictions_valid = self.predict(validation_input_set)
        print("Accuracy of Validation Set: " + str(accuracy(validation_label_set, predictions_valid) * 100))
        # ------------------------------------------------------------


    # Function to train the model with given patterns
    def train_MLP (self, test_input_set, test_label_set):
        print("Training for " + str(len(test_input_set)) + " inputs:")
        cumulative_loss = 100

        for epoch in range(self.max_epochs):
            if abs(cumulative_loss) < self.cumulative_acceptable_loss:
                break

            cumulative_loss = 0

            for i in range(len(test_input_set)):
                # Step 1: Apply Input Pattern
                self.apply_InputPattern(test_input_set[i])

                # Step 2: Forward Propagation - Propagate the signal forward through the network
                self.forward_propagation()

                # Step 3: Output Error Measure
                loss = self.loss.loss_function(test_label_set[i], self.actual_out)
                cumulative_loss += abs(loss)

                # Step 4: Backprapagate the Error Signals - if 'Error-Measure' is not acceptable
                # if abs(loss) > 0.000001:
                self.back_propagation(test_label_set[i])

            print("Epoch: " + str(epoch))
            print("Cumulative Loss: " + str(cumulative_loss))


    # For Testing and Validation Purpose
    def predict (self, test_input_set):
        print("Prediciting for " + str(len(test_input_set)) + " inputs:")
        predictions = None

        for i in range(len(test_input_set)):
            # Step 1: Apply Input Pattern
            self.apply_InputPattern(test_input_set[i])

            # Step 2: Forward Propagation - Propagate the signal forward through the network
            self.forward_propagation()

            curr_pred = np.array([np.round(self.actual_out)])

            if i == 0:
                predictions = curr_pred
            else:
                predictions = np.concatenate((predictions, curr_pred))

        return predictions


    # Step 1 - Apply Input Pattern
    def apply_InputPattern (self, input_features):
        # -------------- Assign outputs in Input Layer -----------------
        input_features = np.append(input_features, -1)
        self.actual_in = np.array(input_features, dtype=float)
        # ------------------------------------------------------------


    # Step 2 - Forward Propagation
    def forward_propagation (self):
        # -------------- Find outputs in Hidden Layer -----------------
        # Find sum (same as the dot product)
        pre_activation_hidden = np.dot(self.w_IH, np.transpose(self.actual_in))

        # Apply Activation Function
        self.actual_hidden = self.activation.activation_function(pre_activation_hidden, self.hidden_activation)
        # ------------------------------------------------------------

        # -------------- Find outputs in Output Layer -----------------
        # Find sum (same as the dot product)
        pre_activation_out = np.dot(self.w_HO, np.transpose(np.append(self.actual_hidden, -1)))

        # Apply Activation Function
        self.actual_out = self.activation.activation_function(pre_activation_out, self.out_activation)
        # ------------------------------------------------------------


    # Step 4 - Backward Propagation
    def back_propagation (self, expected):
        # -------------- Find Error Signals in Output Layer & Changed Weights -----------------
        # Array to store error signals at output-layer
        self.es_out = self.loss.loss_function_derivative(expected, self.actual_out) * self.activation.activation_function_derivative(self.actual_out, self.out_activation)

        # Weight Update
        self.w_HO += self.n * np.dot(np.transpose([self.es_out]), [np.append(self.actual_hidden, -1)])
        # ------------------------------------------------------------

        # -------------- Find Error Signals in Hidden Layer & Changed Weights -----------------
        # Array to store error signals at hidden-layer
        dot_prod = np.dot(np.transpose(self.w_HO), self.es_out)
        self.es_hidden = self.activation.activation_function_derivative(self.actual_hidden, self.hidden_activation) * dot_prod[0 : len(dot_prod) - 1]

        # Weight Update
        self.w_IH += self.n * np.dot(np.transpose([self.es_hidden]), [self.actual_in])
        # ------------------------------------------------------------
    


    # Function to save the model as an object - on disk
    def saveModel(self, file_name):
        with open(file_name, 'wb') as pk_object:
            pickle.dump(self, pk_object, pickle.HIGHEST_PROTOCOL)

    # Function to read the model - already available on disk
    def retrieveModel(self, file_name):
        mlp_model = None

        with open(file_name, 'rb') as pk_object:
            mlp_model = pickle.load(pk_object)

        return mlp_model
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------





# -------------------------------------------------------------------------------------------------
#       Class - to group all Activation Functions
# -------------------------------------------------------------------------------------------------
class Activation ():
    def activation_function (self, input, f_name):
        if (f_name == "Sigmoid"):
            return self.sigmoid(input)
        elif (f_name == "SoftMax"):
            return self.SoftMax(input)
        
    def activation_function_derivative (self, output, f_name):
        if (f_name == "Sigmoid"):
            return self.sigmoid_derivative(output)
        elif (f_name == "SoftMax"):
            return self.SoftMax_derivative(output)


    def sigmoid (self, input):
        return np.transpose(1 / (1 + np.exp(-np.transpose(input))))

    def sigmoid_derivative (self, output):
        temp = np.transpose(output)
        return np.transpose(temp * (1 - temp))


    def SoftMax (self, input):
        temp = np.transpose(input)

        num = np.exp(temp)
        sum = np.sum(np.exp(temp))

        return np.transpose(num/sum)

    def SoftMax_derivative (self, output):
        temp = np.transpose(output)
        return np.transpose(temp * (1 - temp))
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------




# -------------------------------------------------------------------------------------------------
#       Class - to group all Loss Functions
# -------------------------------------------------------------------------------------------------
class Loss ():
    def __init__ (self, f_name):
        self.f_name = f_name

    def loss_function (self, expected, actual):
        if (self.f_name == "Cross Entropy Loss"):
            return self.crossEntropyLoss(expected, actual)
        elif (self.f_name == "Least Square Error Loss"):
            return self.meanSquareErrorLoss(expected, actual)

    def loss_function_derivative (self, expected, actual):
        if (self.f_name == "Cross Entropy Loss"):
            return self.crossEntropyLoss_derivative(expected, actual)
        elif (self.f_name == "Least Square Error Loss"):
            return self.meanSquareErrorLoss_derivative(expected, actual)
        

    def crossEntropyLoss (self, expected, actual):
        actual_T = np.transpose(actual)
        expected_T = np.transpose(expected)
        loss = -np.sum(expected_T * np.log(actual_T)) / len(expected)

        return loss

    def crossEntropyLoss_derivative (self, expected, actual):
        return np.transpose(np.transpose(expected) - np.transpose(actual))


    def leastSquareErrorLoss (self, expected, actual):
        return float(- ((1/2) * pow((expected - actual), 2)))

    def leastSquareErrorLoss_derivative (self, expected, actual):
        return np.transpose(np.transpose(expected) - np.transpose(actual))
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


