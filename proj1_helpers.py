# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def split_data(tx, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    ind = np.random.permutation(len(y))
    limit = int(np.floor(ratio * len(y)))
    train_tx = tx[ind[:limit]]
    train_y = y[ind[:limit]]
    test_tx = tx[ind[limit:]]
    test_y = y[ind[limit:]]
    return train_tx, train_y, test_tx, test_y

def read_csv_headers(data_path):
    with open(data_path, 'r') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

    return fieldnames

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::20]
        input_data = input_data[::20]
        ids = ids[::20]

    return yb, input_data, ids

#Create polynomial terms for each column of tx 
def get_polynomial(tx,poly_term):
	# Author: BN
	# Date: 18/10/2018
	tx_temp = tx.copy()
	tx_return = tx.copy()
	for i in range(poly_term-1):
		tx_temp = np.multiply(tx_temp,tx)
		tx_return = np.c_[ tx_return, tx_temp ]
	return tx_return

def get_data_by_type(input_data, type_tag):
    """Returns a sub-set of the input data based on tag - PRI or DER"""
    if type_tag == 'PRI':
        input_data_ss = input_data[:, :13]
    elif type_tag == 'DER':
        input_data_ss = input_data[:, 13:]
    return input_data_ss

# Standardize the training data using Z score standardization
def standardize_training(x):
	# Author : OT
	# Date: 10/10/2018
	mean_x = np.mean(x,axis= 0)
	x = x - mean_x
	std_x = np.std(x,axis= 0)
	x = x / std_x
	return x, mean_x, std_x
	
# Standardize the test data using the mean and SD of the training data
def standardize_test(x,mean_x,std_x):
	"""Standardize the original data set."""
	x = x - mean_x
	x = x / std_x
	return x

# Standardize data using min-max standardization
def standardize_minmax(tx):
	n_col = tx.shape[0]
	tx_return = tx.copy()
	for col in range(n_col):
		tx_return[:,col] = (tx[:,col] - tx[:,col].amin())/(tx[:,col].amax() - tx[:,col].amin())
	return tx_return

# Remove outliers from the dataset. i.e. points which are more than 2 SD away.
def remove_outliers(tx,y,threshold):
    #Author : BN 
	#Date: 17/10/2018
    shape_tx = np.shape(tx)
    n_columns = shape_tx[1]
    for i_col in range(n_columns):
        y = y[np.where(np.abs(tx[:,i_col]) < threshold)]
        tx = tx[np.where(np.abs(tx[:,i_col]) < threshold)]       
    return tx, y


#remove columns with large 999
def remove_columns_invalid(input_data, thresh):
	# Author : SS
	# Modification : BN 
	# Modified Date : 17/10/2018
    input_data_ss = input_data
    ct = 0
    for column in input_data.T:
        num_invalid = len(column[np.where(column == -999)])
        # print(float(num_invalid) / len(column) * 100)
        if float(num_invalid) / len(column) * 100 > thresh:
            input_data_ss = np.delete(input_data_ss, ct, 1)
            # print("Shape:", np.shape(input_data_ss))
            # print('ct is:',ct)            
#         ct += 1
        else: ct += 1 # BN edit to account for renumbering of columns
    return input_data_ss

#Generates class predictions given weights, and a test data matrix
def predict_labels(weights, data):
    
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

#Verify the predicted y against the test y for cross validation data sets 
def verify_prediction(y_pred, y_test):
    #Author : BN 
	#Date: 17/10/2018
    y_diff = y_pred - y_test
    nFalse = len(y_diff[y_diff !=0 ])
    # accuracy = 1 - np.sum(y_diff)/len(y_diff)
    accuracy = 1 - nFalse/len(y_diff)
    return accuracy
	
def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

if __name__ == "__main__":

    data_path = "/Users/sandra/Downloads/all/train.csv"
    print(read_csv_headers(data_path))
    yb, input_data, ids = load_csv_data(data_path, sub_sample=True)
    idss = get_data_by_type(input_data, "PRI")
    print(np.shape(idss))
    idss1 = remove_columns_invalid(idss, 50)
    print(np.shape(idss1))
