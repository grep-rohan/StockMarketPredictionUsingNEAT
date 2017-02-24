import time

import quandl

start = time.clock()

sensex = None
exchange = None


def main():
    global sensex, exchange
    # if len(sys.argv) != 3:
    #     print('usage: get_historical_data.py [from (yyyy-mm-dd)] [to (yyyy-mm-dd)]')
    #     sys.exit()
    # from_date = sys.argv[1]
    # to_date = sys.argv[2]
    from_date = '1991-02-04'
    to_date = '2017-02-19'
    sensex = quandl.get('BSE/SENSEX', authtoken="o7Xx9kF1M67g-DyENmfZ", start_date=from_date, end_date=to_date)
    exchange = quandl.get('FRED/DEXINUS', authtoken="o7Xx9kF1M67g-DyENmfZ", start_date=from_date, end_date=to_date)

    print(len(exchange) - len(sensex))


main()

print('Time Taken:', str(time.clock() - start))
del start
