import numpy as np
from l2_distance import l2_distance
from utils import load_test,load_train,load_train_small,load_valid
from plot_digits import plot_digits
import pylab as plt


def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """
    # TODO call l2_distance to compute distance between valid data and train data
    dist = l2_distance(valid_data.T,train_data.T)

    # TODO sort the distance to get top k nearest data
    valid_labels = np.array([])
    for i in range(dist.shape[0]):

        element = np.c_[dist[i].T,train_labels]  # the distance to each traning data + training data labels
        element = element[element[:,0].argsort()]  # sort by distance (column 0)
        element = element[:,1]  # choose column[1]
        element = element[:k]  # The K distances that nearest to sample
        if i == 0:
            valid_labels = [element]  # adding one sample distance result list to a total sample list
        else:
            valid_labels = np.r_[valid_labels,[element]]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels




#print train_data, train_lable
#plot_digits(train_data)

#print(train_data.shape,train_lable.shape,valid_data.shape)
train_data, train_label = load_train_small()
classification_rate = []
for i in [1,3,5,7,9]:
    '''
    Using Validation data(total 25 samples for each class) to calibrate value K
    '''
    valid_data, valid_label = load_valid()
    result = run_knn(i,train_data,train_label,valid_data).T   #return one cloumn of validation data
    result = (result[0])
    valid_label = (valid_label.T.astype(np.int)[0])
    compare = abs(result-valid_label)
    wrong_answer = sum(abs(result-valid_label))
    length = len(abs(result-valid_label))
    classification_rate.append([float(1) - float(wrong_answer)/float(length),i])

classification_set = np.array(classification_rate)  # Set classification_set to Numpy
# Plot validation classification vs value k
plt.plot(classification_set.T[1],classification_set.T[0],)
print classification_rate
plt.ylim(0.9,1)
plt.xlabel('K')
plt.ylabel('classification rate')
plt.show()

# Choose K = 5
test_data, test_lables = load_test()
for k in [3,5,7 ]:
    result = run_knn(k,train_data,train_label,test_data).T
    print(result)
plt.plot(result)
plt.ylim(0.9,1)
plt.xlabel('K')
plt.ylabel('classification rate')
plt.show()