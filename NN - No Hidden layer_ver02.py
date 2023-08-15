# Soheil Abadifard
# 22101026
# HW 2
# NN - no hidden layer

import matplotlib.pyplot as plt
import numpy as np


def normalizer(x):
    return np.mean(x), np.std(x), (x - np.mean(x)) / np.std(x)


def denormalizer(x, mean, std):
    x = (x * std) + mean
    return x


def data_splitter(x):
    """ function for Splitting the data to the X and Y"""
    x_x = x[:, 0].reshape(x.shape[0], 1)
    x_y = x[:, 1].reshape(x.shape[0], 1)
    return x_x, x_y


def plot_function(x, y, x_label, y_label, text, name):
    plt.plot(x, y, label="MSE")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    txt = text
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    # plt.savefig(name, bbox_inches='tight')
    plt.show()


class nn_regressor(object):

    def __init__(self, learning_rate=0.6, beta=0.9, momentum=False):
        self.momentum = momentum
        self.error = None
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight = (((np.random.rand(1, 1)) * 2) - 1)
        self.bias = (((np.random.rand(1, 1)) * 2) - 1)
        self.momentum_w = 0
        self.momentum_b = 0

    def fw(self, independent_variables):
        return np.dot(independent_variables, self.weight) + self.bias

    def bp(self, independent_variables, dependent_variables, predicted_values):

        if self.momentum:
            self.error = (2 * (predicted_values - dependent_variables)) / independent_variables.shape[0]
            # delta_weight = (independent_variables.T.dot(self.error) * self.learning_rate) /
            # independent_variables.shape[0]
            delta_weight = (np.dot(independent_variables.T, self.error) * self.learning_rate) + (
                    self.momentum_w * self.beta)
            self.momentum_w = delta_weight
            delta_bias = np.sum(self.error * self.learning_rate) + (self.momentum_b * self.beta)
            self.momentum_b = delta_bias

            self.weight -= delta_weight
            self.bias -= delta_bias

        else:
            self.error = (2 * (predicted_values - dependent_variables)) / independent_variables.shape[0]
            # delta_weight = (independent_variables.T.dot(self.error) * self.learning_rate) /
            # independent_variables.shape[0]
            delta_weight = np.dot(independent_variables.T, self.error) * self.learning_rate
            delta_bias = np.sum(self.error * self.learning_rate)

            self.weight -= delta_weight
            self.bias -= delta_bias

    def train(self, indep, dep):
        output = self.fw(indep)
        self.bp(indep, dep, output)

    def predict(self, x):
        return self.fw(x)

    def loss(self, x, y):
        return np.mean(np.square(y - self.predict(x)))


train1 = np.loadtxt(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/HWs/HW2/Data/train1.txt")
train2 = np.loadtxt(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/HWs/HW2/Data/train2.txt")
test1 = np.loadtxt(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/HWs/HW2/Data/test1.txt")
test2 = np.loadtxt(
    "/HDD/F/University/Bilkent University/Second Semester/CS 550 Machine Learning/HWs/HW2/Data/test2.txt")

train1_x, train1_y = data_splitter(train1)
test1_x, test1_y = data_splitter(test1)
train2_x, train2_y = data_splitter(train2)
test2_x, test2_y = data_splitter(test2)

# ---------------------------------------------------------------------------------------------------------------------
# first we check the nn for original data (train 1) and try to find the best epoch number / Here I use Batch method


# same process with stochastic approach


# ---------------------------------------------------------------------------------------------------------------------
# then  we check the nn for normalized data (train 1) and try to find the best epoch number / Batch Approach

train1_x_mean, train1_x_std, train1_x_normalized = normalizer(train1_x)
train1_y_mean, train1_y_std, train1_y_normalized = normalizer(train1_y)

epoch_list = []
error_list = []
for i in range(1, 20):
    nn_for_normalized_data = nn_regressor(learning_rate=0.6)
    epoch = i
    epoch_list.append(epoch)
    print("value for maximum Epoch %s " % i)

    while epoch > 0:
        print("Loss Train: \n" + str(nn_for_normalized_data.loss(train1_x_normalized, train1_y_normalized)))
        print("\n")
        nn_for_normalized_data.train(train1_x_normalized, train1_y_normalized)
        epoch -= 1
    error_list.append(nn_for_normalized_data.loss(train1_x_normalized, train1_y_normalized))

print("the minimum loss is : ", min(error_list), " and the index is : ", error_list.index(min(error_list)))

plot_function(epoch_list, error_list, 'Epochs', 'MSE', "Epoch VS MSE - Normalized Data (train set 1) - Batch Mode",
              'Epoch VS MSE - Normalized Data (train set 1) - Batch Mode')

# Same Process with Stochastic Approach

# same process with momentum / Batch

# Same Process with momentum / Stochastic Approach

# ---------------------------------------------------------------------------------------------------------------------
# predicting using same train set to check the accuracy (train set 1)

train1_prediction = nn_for_normalized_data.predict(train1_x_normalized)

plt.plot(train1_x, denormalizer(train1_prediction, train1_y_mean, train1_y_std))
plt.scatter(train1_x, train1_y, label="Training Data")
plt.plot(train1_x, denormalizer(train1_prediction, train1_y_mean, train1_y_std), label="Predicted")
txt = "No hidden layer 1st train set - results on Train set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('No hidden layer 1st train set - results on Train set', bbox_inches='tight')
plt.show()

test1_x_normalized = (test1_x - train1_x_mean) / train1_x_std

test1_prediction = nn_for_normalized_data.predict(test1_x_normalized)
test_loss_value = nn_for_normalized_data.loss(test1_x_normalized, ((test1_y * train1_y_std) + train1_y_mean))
print("Loss Test: \n" + str(test_loss_value))

plt.plot(test1_x, denormalizer(test1_prediction, train1_y_mean, train1_y_std))
plt.scatter(test1_x, test1_y, label="Test Data")
plt.plot(test1_x, denormalizer(test1_prediction, train1_y_mean, train1_y_std), label="Predicted")
txt = "No hidden layer 1st train set - results on Test set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('No hidden layer 1st train set - results on Test set', bbox_inches='tight')
plt.show()

# =====================================================================================================================

# for second data set

train2_x_mean, train2_x_std, train2_x_normalized = normalizer(train2_x)
train2_y_mean, train2_y_std, train2_y_normalized = normalizer(train2_y)

epoch_list = []
error_list = []
for i in range(1, 300):
    nn_for_normalized_data = nn_regressor(learning_rate=0.6, momentum=True)
    epoch = i
    epoch_list.append(epoch)
    print("value for maximum Epoch %s " % i)

    while epoch > 0:
        print("Loss Train: \n" + str(nn_for_normalized_data.loss(train2_x_normalized, train2_y_normalized)))
        print("\n")
        nn_for_normalized_data.train(train2_x_normalized, train2_y_normalized)
        epoch -= 1
    error_list.append(nn_for_normalized_data.loss(train2_x_normalized, train2_y_normalized))

print("the minimum loss is : ", min(error_list), " and the index is : ", error_list.index(min(error_list)))

plot_function(epoch_list, error_list, 'Epochs', 'MSE', "Epoch VS MSE - Normalized Data (train set 2) - Batch Mode",
              'Epoch VS MSE - Normalized Data (train set 2) - Batch Mode')

train2_prediction = nn_for_normalized_data.predict(train2_x_normalized)

plt.plot(train2_x, denormalizer(train2_prediction, train2_y_mean, train2_y_std))
plt.scatter(train2_x, train2_y, label="Training Data")
plt.plot(train2_x, denormalizer(train2_prediction, train2_y_mean, train2_y_std), label="Predicted")
txt = "No hidden layer 2nd train set - results on Train set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('No hidden layer 2nd train set - results on Train set', bbox_inches='tight')
plt.show()

test2_x_normalized = (test2_x - train2_x_mean) / train2_x_std

test_loss_value = nn_for_normalized_data.loss(test2_x_normalized, ((test2_y * train2_y_std) + train2_y_mean))
print("Loss Test: \n" + str(test_loss_value))

test2_prediction = nn_for_normalized_data.predict(test2_x_normalized)
plt.plot(test2_x, denormalizer(test2_prediction, train2_y_mean, train2_y_std))
plt.scatter(test2_x, test2_y, label="Test Data")
plt.plot(test2_x, denormalizer(test2_prediction, train2_y_mean, train2_y_std), label="Predicted")
txt = "No hidden layer 2nd train set - results on Test set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('No hidden layer 2nd train set - results on Test set', bbox_inches='tight')
plt.show()
