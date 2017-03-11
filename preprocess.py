"""Contains functions to retrieve, scale, split and visualize SENSEX and USD/INR exchange rate historical data."""
import os
import time
from math import floor

import matplotlib.pyplot as plt
import pandas as pd
import quandl


def retrieve(from_date='2003-07-14', to_date='2017-02-19'):
    """
    Retrieve BSE SENSEX and USD/INR exchange rate historical data between the two dates passed as arguments.

    :param from_date: Date from which data is to be retrieved. Format: yyyy-mm-dd, dtype: str
    :param to_date: Date to which data is to be retrieved. Format: yyyy-mm-dd, dtype: str
    :return: Data.
    """
    start_time = time.clock()

    data = None

    # checking if data stored locally
    if os.path.isfile('data'):
        print('Found data locally')
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
            to_date = '2017-02-19'

        print('Retrieving data from quandl')

        sensex = quandl.get('YAHOO/INDEX_BSESN', authtoken="o7Xx9kF1M67g-DyENmfZ", start_date=from_date,
                            end_date=to_date)
        exchange = quandl.get('FRED/DEXINUS', authtoken="o7Xx9kF1M67g-DyENmfZ", start_date=from_date, end_date=to_date)

        data = pd.concat([sensex, exchange], axis=1)  # merging both dataframes
        data.rename(columns={'VALUE': 'Exchange'}, inplace=True)  # renaming exchange column
        data.fillna(method='pad', inplace=True)  # filling missing values with previous values

        data.to_pickle('data')  # storing dataframe in file

    print('Data retrieved in ', time.clock() - start_time, 's')

    return data


def scale(data):
    """
    Scale the data between 0 and 1.

    :return: Scaled data.
    """
    print('Scaling data')

    data_scaled = (data - data.min()) / (data.max() - data.min())

    return data_scaled


def split(data, train_percent=.9):
    """
    Split data into training and testing data.

    :param data: The data to be split.
    :param train_percent: Percentage of data which is training.
    :return: Tuple containing training and testing data.
    """
    train_data = data.iloc[0:floor(train_percent * len(data))]
    test_data = data.iloc[floor(train_percent * len(data)):]

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


def descale(data_scaled, data_original):
    """
    Descale scaled data.

    :param data_scaled: Scaled data which is to be descaled.
    :param data_original: Original data before scaling.
    :return: Descaled data.
    """
    data_descaled = data_scaled * (data_original['Close'].max() - data_original['Close'].min()) + \
                    data_original['Close'].min()

    return data_descaled
