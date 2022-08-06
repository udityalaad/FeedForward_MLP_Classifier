from mlp import MLP_Neural_Nw
import numpy as np
import acc_calc
import test_mlp

# -------------------------------------------------------------------------------------------------
#       Main ()
# -------------------------------------------------------------------------------------------------
def main ():
    # # n  ->  learning rate
    # # size_hidden  ->  No. of neurons in the hidden layer
    # # test_validation_division  ->  test_size / total_sample_size
    
    # mlp = MLP_Neural_Nw(size_input_features = 784, size_hidden = 100, size_out = 4, n = 0.2,
    #                     activations = ["Sigmoid", "SoftMax"], loss_function="Cross Entropy Loss",
    #                     max_epochs = 30, test_validation_division = 9/10, individual_acceptable_loss = 0.001)
    # mlp.build_Model('train_data.csv', 'train_labels.csv')
    # mlp.saveModel('model_q4_a.pk1')

    pred = test_mlp.test_mlp('train_data.csv')
    label_data = np.loadtxt('train_labels.csv', delimiter=",", dtype=float)
    print("Accuracy: " + str(acc_calc.accuracy(label_data , pred) * 100))


if __name__ == "__main__":
	main()
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------