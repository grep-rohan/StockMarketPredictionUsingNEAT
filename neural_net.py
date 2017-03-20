import datetime as dt
import time
import warnings

from pandas import Series, DataFrame
from sklearn.neural_network import MLPRegressor

import preprocess

warnings.filterwarnings("ignore")

start = time.clock()

attributes = ('Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close', 'Exchange')
days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri')

if __name__ == '__main__':
    data = preprocess.retrieve()
    train, test = preprocess.split(data, train_percent=.6)
    train = preprocess.scale(train)
    test = preprocess.scale(test)
    inputs = [[train.iloc[index][attribute] for attribute in attributes] +
              [1 if dt.datetime.strptime(str(train.iloc[index].name)[:-9], '%Y-%m-%d').strftime('%a') == day else 0
               for day in days]
              for index in range(len(train) - 1)]
    outputs = [train.iloc[i + 1]['Close'] for i in range(len(train) - 1)]
    test_inputs = [[test.iloc[index][attribute] for attribute in attributes] +
                   [1 if dt.datetime.strptime(str(test.iloc[index].name)[:-9], '%Y-%m-%d').strftime('%a') == day else 0
                    for day in days]
                   for index in range(len(test) - 1)]
    test_outputs = [test.iloc[i + 1]['Close'] for i in range(len(test) - 1)]

    neural_net = MLPRegressor(hidden_layer_sizes=(7,), activation='tanh', solver='adam',
                              learning_rate='adaptive', max_iter=2000, verbose=True)
    neural_net.fit(inputs, outputs)
    preprocess.visualize(Series(neural_net.loss_curve_))

    daily = DataFrame(columns=('Actual Daily', 'Predicted Daily'))
    count = 0
    cost = 0
    for ip, op in zip(test_inputs, test_outputs):
        output = neural_net.predict(ip)
        daily.loc[count] = [op, output[0]]
        count += 1
        cost += (output[0] - op) ** 2

    print('Test Cost:', -.5 * cost)

    preprocess.visualize(daily)

print(time.clock() - start)
