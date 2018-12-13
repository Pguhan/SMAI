'''
    A framework which takes in a CSV(arg 1) with last column as the label and runs the ML clasifiers.
'''

from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import sys

try:
    input_file = sys.argv[1]
    input_data = np.genfromtxt(input_file, delimiter=',')
    print input_data
except:
    print "File(arg1) not found or not able to open."
    exit()

#randomly shuffle the rows and pick the top 80% for train and test on rest 20%
np.random.shuffle(input_data)

input_X = input_data[:,:-1]
#PCA done here to 8 top features
pca = PCA(n_components = 12)
input_X = pca.fit_transform(input_X)
input_label = input_data[:,-1]
input_label = input_label.astype(int)

total_num_rows = input_data.shape[0]
num_rows = (int)(total_num_rows * 0.8)

train_X = input_X[:num_rows,:]
train_label = input_label[:num_rows]
test_X = input_X[num_rows:,:]
test_label = input_label[num_rows:]

run_svm = svm.SVC()
run_svm.fit(train_X, train_label)
print "Mean Accuracy on SVM = ", run_svm.score(test_X, test_label)
