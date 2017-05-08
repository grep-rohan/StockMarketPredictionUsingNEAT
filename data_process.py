"""Contains functions to retrieve, scale, split and visualize SENSEX and USD/INR exchange rate historical data."""
import os
import pickle
import time
from math import floor

import matplotlib.pyplot as plt
import pandas as pd
import quandl


def retrieve(from_date='1999-02-01', to_date='2017-04-28'):
    """
    Retrieve BSE SENSEX and USD/INR exchange rate historical data between the two dates passed as arguments.

    :param from_date: Date from which data is to be retrieved. Format: yyyy-mm-dd, dtype: str
    :param to_date: Date to which data is to be retrieved. Format: yyyy-mm-dd, dtype: str
    :return: Data.
    """
    # checking if data stored locally
    if os.path.isfile('data'):
        print('\nLoading cached data...')
        data = pd.read_pickle('data')
    else:
        # validating dates
        try:
            time.strptime(from_date, '%Y-%m-%d')
            time.strptime(to_date, '%Y-%m-%d')
        except ValueError:
            print('Invalid date(s)!\nThe dates should be string and the format should be yyyy-mm-dd\n\n'
                  'Using default values (from_date = 2003-07-14, to_date = 2017-02-19\n')
            from_date = '2003-07-14'
            to_date = '2017-03-05'

        print('\nRetrieving data from quandl...')

        api_key = open('quandl_api_key.txt', 'r').read()  # the api key is stored as plain text in quandl_api_key.txt

        # sensex = quandl.get('YAHOO/INDEX_BSESN', authtoken=api_key, start_date=from_date, end_date=to_date)
        # exchange = quandl.get('FRED/DEXINUS', authtoken=api_key, start_date=from_date, end_date=to_date)

        # data = pd.concat([sensex, exchange], axis=1)  # merging both dataframes
        # data.rename(columns={'Value': 'Exchange'}, inplace=True)  # renaming exchange column
        # data.fillna(method='pad', inplace=True)  # filling missing values with previous values
        # data = data.drop('Volume', axis=1)  # dropping volume

        data = quandl.get('BSE/BSE500', authtoken=api_key, start_date=from_date, end_date=to_date)
        del data['Open'], data['High'], data['Low']

        print('\nCaching data...')
        data.to_pickle('data')  # saving data to file

    return data


def scale(data):
    """
    Scale the data between -1 and 1.

    :param data: The data to be scaled. Data type : Pandas DataFrame
    :return: Scaled data. Data Type : Pandas DataFrame
    """
    return (data - data.mean()) / (data.max() - data.min())


def split(data, train_size):
    """
    Split data into training and testing data.

    :param data: The data to be split.
    :param train_size: Percentage of data which is training.
    :return: Tuple containing training and testing data.
    """
    train_data = data.iloc[0:floor(train_size * len(data))]
    test_data = data.iloc[floor(train_size * len(data)):]

    return train_data, test_data


def visualize(data, style='ggplot', graph_type='line'):
    """
    Visualize data.

    :param data: Data to visualize. (Pandas dataframe)
    :param style: matplotlib style. def = 'ggplot'
    :param graph_type: Graph type ('line' or 'area'). def = 'line'
    :return: None
    """
    try:
        plt.style.use(style)
    except OSError:
        print('\nInvalid style!\nUsing ggplot\n')
        plt.style.use('ggplot')
    if graph_type == 'line':
        data.plot()
    elif graph_type == 'area':
        data.plot.area(stacked=False)
    else:
        print('\nInvalid type!\nUsing line')
        data.plot()
    plt.show()


def strip(data):
    """
    Strip data as per requirement.
    
    :param data: Data to be stripped. 
    :return: Stripped data.
    """
    if len(data) % 30 != 0:
        n = len(data) + (30 - len(data) % 30)
        n -= 30
    else:
        n = len(data) - 30
    return data.iloc[:n + 1, :]


def preprocess():
    if not os.path.isfile('preprocessed_data'):
        data = retrieve()  # retrieve data

        print('\nPreprocessing data...')
        data = scale(data)  # scale
        train, test = split(data, train_size=.9)  # split into training and testing data
        train = strip(train)
        test = strip(test)

        # x_train = [[train.iloc[i][attribute] for i in range(index, index + 30) for attribute in attributes]
        #            for index in range(0, len(train) - 1, 30)]
        # y_train = [train.iloc[i]['Close'] for i in range(30, len(train), 30)]
        # x_test = [[test.iloc[i][attribute] for i in range(index, index + 30) for attribute in attributes]
        #           for index in range(0, len(test) - 1, 30)]
        # y_test = [test.iloc[i]['Close'] for i in range(30, len(test), 30)]

        x_train = [[train.iloc[i]['Close'] for i in range(index, index + 30)] for index in range(0, len(train) - 30)]
        y_train = [train.iloc[i]['Close'] for i in range(30, len(train))]
        x_test = [[test.iloc[i]['Close'] for i in range(index, index + 30)] for index in range(0, len(test) - 30)]
        y_test = [test.iloc[i]['Close'] for i in range(30, len(test))]

        print('\nCaching preprocessed data...\n')
        pickle.dump((x_train, x_test, y_train, y_test), open('preprocessed_data', 'wb'))
    else:
        print('\nLoading cached preprocessed data...\n')
        x_train, y_train, x_test, y_test = pickle.load(open('preprocessed_data', 'rb'))

    return x_train, x_test, y_train, y_test


def descale(data_scaled):
    """
    Descale scaled data.

    :param data_scaled: Scaled data which is to be descaled.
    :return: Descaled data.
    """
    data_original = retrieve()

    data_descaled = data_scaled * (data_original['Close'].max() - data_original['Close'].min()) \
                    + data_original['Close'].mean()

    return data_descaled
