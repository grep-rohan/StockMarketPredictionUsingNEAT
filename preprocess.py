import quandl


def retrieve(from_date, to_date):
    """
    Used to retrieve BSE SENSEX and USD/INR historical data between the 2 dates passed as arguments.

    :param from_date: Date from which data is to be retrieved. Format: yyyy-mm-dd, dtype: str
    :param to_date: Date to which data is to be retrieved. Format: yyyy-mm-dd, dtype: str
    :return: Tuple containing sensex dataframe at index 0 and exchange dataframe at index 1
    """
    sensex = quandl.get('YAHOO/INDEX_BSESN', authtoken="o7Xx9kF1M67g-DyENmfZ", start_date=from_date, end_date=to_date)
    exchange = quandl.get('FRED/DEXINUS', authtoken="o7Xx9kF1M67g-DyENmfZ", start_date=from_date, end_date=to_date)

    return sensex, exchange


def normalize():
    """
    Used to normalize sensex and exchange dataframes.

    :return: Tuple cotaining normalized sensex and exchange dataframes
    """
    sensex, exchange = retrieve('2003-07-14', '2017-02-17')

    sensex_norm = (sensex - sensex.mean()) / (sensex.max() - sensex.min())
    exchange_norm = (exchange - exchange.mean()) / (exchange.max() - exchange.min())

    return sensex_norm, exchange_norm
