import numpy as np

from acc_calc import accuracy
import mlp

STUDENT_NAME = 'Uditya Laad, Dakota Wong, Himalaya Sharma'
STUDENT_ID = '20986041, 20985951, 20971031'


def test_mlp(data_file):
	# Load the test set
	# START
	test_input = np.loadtxt(data_file, delimiter=",", dtype=float)
    # END


	# Load your network
	# START
	object_file = 'model_q4_a.pk1'
	model = mlp.MLP_Neural_Nw()
	model = model.retrieveModel(object_file)
	# END

	# Predict test set - one-hot encoded
	y_pred = model.predict(test_input)
	return y_pred

'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''