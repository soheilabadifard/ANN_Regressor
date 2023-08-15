# Soheil Abadifard
# 22101026
# HW2
# ANN - One layer
import matplotlib.pyplot as plt
import numpy as np


def normalizer_train(x):
    return np.mean(x), np.std(x), (x - np.mean(x)) / np.std(x)


def normalizer_test(x, mean, std):
    return (x - mean) / std


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


class ann(object):

    def __init__(self, number_of_nodes_in_hidden_layer=8, learning_rate=0.3, beta=0.88, momentum=False,
                 activation_function="sigmoid"):
        self.hidden_layer_derivative = None
        self.result = None
        self.hidden_layer = None
        self.product = None
        self.error = None
        self.product = None

        self.number_of_hidden_layer = number_of_nodes_in_hidden_layer
        self.learning_rate = learning_rate
        self.beta = beta
        self.momentum = momentum
        self.activation_function = activation_function

        self.weight_1 = (((np.random.rand(1, self.number_of_hidden_layer)) * 2) - 1)
        self.bias_1 = (((np.random.rand(1, self.number_of_hidden_layer)) * 2) - 1)
        self.weight_2 = (((np.random.rand(self.number_of_hidden_layer, 1)) * 2) - 1)
        self.bias_2 = (((np.random.rand(1, 1)) * 2) - 1)

        self.momentum_w_1 = 0
        self.momentum_w_2 = 0
        self.momentum_b_1 = 0
        self.momentum_b_2 = 0

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        # return self.sigmoid(x) * (1 - self.sigmoid(x))
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_derivative(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def fw(self, independent_variables):
        if self.activation_function == "sigmoid":
            self.product = np.dot(independent_variables, self.weight_1) + self.bias_1
            self.hidden_layer = self.sigmoid(self.product)
            self.result = np.dot(self.hidden_layer, self.weight_2) + self.bias_2
            return self.result
        else:
            self.product = np.dot(independent_variables, self.weight_1) + self.bias_1
            self.hidden_layer = self.relu(self.product)
            self.result = np.dot(self.hidden_layer, self.weight_2) + self.bias_2
            return self.result

    def bp(self, independent_variables, dependent_variables, predicted_values):
        if self.momentum:

            if self.activation_function == "sigmoid":

                self.error = (2 * (predicted_values - dependent_variables)) / independent_variables.shape[0]

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.sigmoid_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((independent_variables.T.dot(
                    self.hidden_layer_derivative)) * self.learning_rate) + self.beta * self.momentum_w_1
                delta_weight_2 = ((self.hidden_layer.T.dot(
                    self.error)) * self.learning_rate) + self.beta * self.momentum_w_2

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0,
                                      keepdims=True) + self.beta * self.momentum_b_1
                delta_bias_2 = np.sum(self.error * self.learning_rate) + self.beta * self.momentum_b_2

                self.momentum_w_1 = delta_weight_1
                self.momentum_w_2 = delta_weight_2
                self.momentum_b_1 = delta_bias_1
                self.momentum_b_2 = delta_bias_2

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2

            else:
                self.error = (2 * (predicted_values - dependent_variables)) / independent_variables.shape[0]

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.relu_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((independent_variables.T.dot(
                    self.hidden_layer_derivative)) * self.learning_rate) + self.beta * self.momentum_w_1
                delta_weight_2 = ((self.hidden_layer.T.dot(
                    self.error)) * self.learning_rate) + self.beta * self.momentum_w_2

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0,
                                      keepdims=True) + self.beta * self.momentum_b_1
                delta_bias_2 = np.sum(self.error * self.learning_rate) + self.beta * self.momentum_b_2

                self.momentum_w_1 = delta_weight_1
                self.momentum_w_2 = delta_weight_2
                self.momentum_b_1 = delta_bias_1
                self.momentum_b_2 = delta_bias_2

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2
        else:
            if self.activation_function == "sigmoid":
                self.error = (2 * (predicted_values - dependent_variables))

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.sigmoid_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((independent_variables.T.dot(self.hidden_layer_derivative)) * self.learning_rate) / \
                                 independent_variables.shape[0]
                delta_weight_2 = ((self.hidden_layer.T.dot(self.error)) * self.learning_rate) / \
                                 independent_variables.shape[0]

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0, keepdims=True) / \
                               independent_variables.shape[0]
                delta_bias_2 = np.sum(self.error * self.learning_rate) / independent_variables.shape[0]

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2
            else:
                self.error = (2 * (predicted_values - dependent_variables))

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.relu_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((independent_variables.T.dot(self.hidden_layer_derivative)) * self.learning_rate) / \
                                 independent_variables.shape[0]
                delta_weight_2 = ((self.hidden_layer.T.dot(self.error)) * self.learning_rate) / \
                                 independent_variables.shape[0]

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0, keepdims=True) / \
                               independent_variables.shape[0]
                delta_bias_2 = np.sum(self.error * self.learning_rate) / independent_variables.shape[0]

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2

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

train1_x_mean, train1_x_std, train1_x_normalized = normalizer_train(train1_x)
train1_y_mean, train1_y_std, train1_y_normalized = normalizer_train(train1_y)

epoch_list = []
error_list = []
for i in range(1, 2):
    nn_for_normalized_data = ann(number_of_nodes_in_hidden_layer=16, learning_rate=0.3, momentum=True)
    epoch = 0
    print("value for maximum Epoch %s " % i)

    while epoch <= 10000:

        if epoch % 10 == 0:
            epoch_list.append(epoch)
            error_list.append(nn_for_normalized_data.loss(train1_x_normalized, train1_y_normalized))
            print("Loss Train: \n" + str(nn_for_normalized_data.loss(train1_x_normalized, train1_y_normalized)))
            print("\n")
        nn_for_normalized_data.train(train1_x_normalized, train1_y_normalized)
        epoch += 1
print("the minimum loss is : ", min(error_list), " and the index is : ", error_list.index(min(error_list)))
plot_function(epoch_list, error_list, 'Epochs', 'MSE', "Epoch VS MSE - Normalized Data (train set 1) - Batch Mode",
              '1-hidden layer-Epoch VS MSE - Normalized Data (train set 1) - Batch Mode')

train1_prediction = nn_for_normalized_data.predict(train1_x_normalized)
x1, train1_prediction = zip(*sorted(zip(train1_x, denormalizer(train1_prediction, train1_y_mean, train1_y_std))))
plt.plot(x1, train1_prediction)
plt.scatter(train1_x, train1_y, label="Training Data")
plt.plot(x1, train1_prediction, label="Predicted")
txt = "one hidden layer 1st train set - results on Train set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('one hidden layer 1st train set - results on Train set - 16 nodes', bbox_inches='tight')
plt.show()
# ================================================================================================================
test1_x_normalized = normalizer_test(test1_x, train1_x_mean, train1_x_std)
test1_y_normalized = normalizer_test(test1_y, train1_y_mean, train1_y_std)

test1_prediction = nn_for_normalized_data.predict(test1_x_normalized)

test_loss_value = np.mean(np.power((test1_y_normalized - test1_prediction), 2))

print("Loss Test: \n" + str(test_loss_value))

x1, test1_prediction = zip(*sorted(zip(test1_x, denormalizer(test1_prediction, train1_y_mean, train1_y_std))))
plt.plot(x1, test1_prediction)

test1_x, test1_y = zip(*sorted(zip(test1_x, test1_y)))

plt.scatter(test1_x, test1_y, label="test Data")

plt.plot(x1, test1_prediction, label="Predicted")
txt = "one hidden layer 1st train set - results on test set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('one hidden layer 1st train set - results on test set - 16 nodes', bbox_inches='tight')
plt.show()

train2_x_mean, train2_x_std, train2_x_normalized = normalizer_train(train2_x)
train2_y_mean, train2_y_std, train2_y_normalized = normalizer_train(train2_y)

thr = 0.05
err = 1
epoch_list = []
error_list = []
for i in range(1, 2):
    nn_for_normalized_data = ann(number_of_nodes_in_hidden_layer=8, learning_rate=0.4, momentum=False)
    epoch = 1
    print("value for maximum Epoch %s " % i)

    while epoch:

        if epoch % 10 == 0:
            epoch_list.append(epoch)
            err = nn_for_normalized_data.loss(train2_x_normalized, train2_y_normalized)
            error_list.append(err)
            print("Loss Train: \n" + str(nn_for_normalized_data.loss(train2_x_normalized, train2_y_normalized)))
            print("epoch :", epoch)
            print("\n")
        if err >= thr:
            nn_for_normalized_data.train(train2_x_normalized, train2_y_normalized)
            epoch += 1
        else:
            epoch = 0

print("the minimum loss is : ", min(error_list), " and the index is : ", error_list.index(min(error_list)))
plot_function(epoch_list, error_list, 'Epochs', 'MSE', "Epoch VS MSE - Normalized Data (train set 2) - Batch Mode",
              '1-hidden layer-Epoch VS MSE - Normalized Data (train set 2) - Batch Mode')

train2_prediction = nn_for_normalized_data.predict(train2_x_normalized)
test_loss_value = np.mean(np.power((train2_y_normalized - train2_prediction), 2))

print("Loss Test: \n" + str(test_loss_value))
x2, train2_prediction = zip(*sorted(zip(train2_x, denormalizer(train2_prediction, train2_y_mean, train2_y_std))))
plt.plot(x2, train2_prediction)
plt.scatter(train2_x, train2_y, label="Training Data")
plt.plot(x2, train2_prediction, label="Predicted")
txt = "one hidden layer 2nd train set - results on Train set - alpha = 0.4, With Momentum"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('one hidden layer 2nd train set - results on Train set - alpha 0-04 with Momentum', bbox_inches='tight')
plt.show()
# ================================================================================================================
test2_x_normalized = normalizer_test(test2_x, train2_x_mean, train2_x_std)
test2_y_normalized = normalizer_test(test2_y, train2_y_mean, train2_y_std)

test2_prediction = nn_for_normalized_data.predict(test2_x_normalized)

test_loss_value = np.mean(np.power((test2_y_normalized - test2_prediction), 2))

print("Loss Test: \n" + str(test_loss_value))

x2, test2_prediction = zip(*sorted(zip(test2_x, denormalizer(test2_prediction, train2_y_mean, train2_y_std))))
plt.plot(x2, test2_prediction)

test2_x, test2_y = zip(*sorted(zip(test2_x, test2_y)))

plt.scatter(test2_x, test2_y, label="test Data")

plt.plot(x2, test2_prediction, label="Predicted")
txt = "one hidden layer 2nd train set - results on test set"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('one hidden layer 2nd train set - results on test set - 16 nodes', bbox_inches='tight')
plt.show()

thr = 0.05
err = 1
epoch_list = []
error_list = []
nn_for_normalized_data1 = ann(number_of_nodes_in_hidden_layer=8, learning_rate=0.25, momentum=True)
epoch = 1

while epoch:

    i = 0
    for instance in train2_x_normalized:
        nn_for_normalized_data1.train(instance, train2_y_normalized[i])
        i = i + 1
    if epoch % 10 == 0:
        err = nn_for_normalized_data1.loss(train2_x_normalized, train2_y_normalized)
        print("Loss Train: \n", err)
        print("epoch :", epoch)
        print("\n")
    error_list.append(err)
    epoch_list = []
    epoch += 1
    if thr >= err:
        epoch = 0

print("Loss Train: \n" + str(nn_for_normalized_data1.loss(train2_x_normalized, train2_y_normalized)))
train2_prediction = nn_for_normalized_data.predict(train2_x_normalized)
test_loss_value = np.mean(np.power((train2_y_normalized - train2_prediction), 2))

print("Loss Test: \n" + str(test_loss_value))
x2, train2_prediction = zip(*sorted(zip(train2_x, denormalizer(train2_prediction, train2_y_mean, train2_y_std))))
plt.plot(x2, train2_prediction)
plt.scatter(train2_x, train2_y, label="Training Data")
plt.plot(x2, train2_prediction, label="Predicted")
txt = "one hidden layer 2nd train set - results on Train set - alpha = 0.4, With Momentum"
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.legend()
plt.savefig('one hidden layer 2nd train set - results on Train set - alpha 0-04 with Momentum', bbox_inches='tight')
plt.show()