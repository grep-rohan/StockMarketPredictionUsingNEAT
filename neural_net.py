import warnings

from pandas import DataFrame, Series
from sklearn.neural_network import MLPRegressor

import data_process

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data_process.preprocess()

    neural_net = MLPRegressor(solver='sgd', verbose=True)
    neural_net.fit(x_train, y_train)

    data_process.visualize(Series(neural_net.loss_curve_), graph_type='area')

    print('\nCalculating training results...')
    training = DataFrame(columns=('Actual', 'Predicted'))
    cost = 0
    for index in range(len(y_train)):
        output = neural_net.predict(x_train[index])
        training.loc[index] = [y_train[index], output[0]]
        cost += (y_train[index] - output[0]) ** 2
    print('\nTraining Cost = %f' % (cost / len(training)))
    training = data_process.descale(training)
    data_process.visualize(training)

    print('\nCalculating testing results...')
    testing = DataFrame(columns=('Actual', 'Predicted'))
    cost = 0
    for index in range(len(y_test)):
        output = neural_net.predict(x_test[index])
        testing.loc[index] = [y_test[index], output[0]]
        cost += (y_test[index] - output[0]) ** 2
    print('\nTesting Cost = %f' % (cost / len(testing)))
    data_process.visualize(testing)
