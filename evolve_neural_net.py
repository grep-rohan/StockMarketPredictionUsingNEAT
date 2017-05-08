"""Evolve neural network for stock market prediction."""
import os
import random
import string

import neat
from pandas import DataFrame
from pastalog import Log

import data_process
import visualize

reporter = neat.StdOutReporter(True)
stats = neat.StatisticsReporter()
attributes = ('Open', 'High', 'Low', 'Close', 'Adjusted Close', 'Exchange')
days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri')
# node_names = {-1: attributes[0], -2: attributes[1], -3: attributes[2], -4: attributes[3], -5: attributes[4],
#              -6: attributes[5], -7: attributes[6], -8: days[0], -9: days[1], -10: days[2], -11: days[3], -12: days[4],
#               0: 'Predicted Close'}
random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
pastslog = Log('http://localhost:8120', random_id)

train = test = data = x_train = y_train = x_test = y_test = None


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        cost = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for index in range(len(y_train)):
            predicted_output = net.activate(x_train[index])
            cost += (predicted_output[0] - y_train[index]) ** 2
        genome.fitness = -cost
    try:
        pass
        gen = reporter.generation - 1
        if gen > 0:
            visualize.draw_net(config=config, genome=stats.best_genome(), view=False)
            pastslog.post('best', value=stats.best_genome().fitness, step=gen)
            pastslog.post('+1 stdev', value=stats.get_fitness_mean()[gen] + stats.get_fitness_stdev()[gen], step=gen)
            pastslog.post('avg', value=stats.get_fitness_mean()[gen], step=gen)
            pastslog.post('-1 stdev', value=stats.get_fitness_mean()[gen] - stats.get_fitness_stdev()[gen], step=gen)
    except:
        print('\npastalog error!\n')


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(reporter)
    p.add_reporter(stats)

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    visualize.draw_net(config, winner, True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    training = DataFrame(columns=('Actual Daily', 'Predicted Daily'))
    cost = 0
    for index in range(len(y_train)):
        output = winner_net.activate(x_train[index])
        training.loc[index] = [y_train[index], output[0]]
        cost += (y_train[index] - output[0]) ** 2
    print('\nTraining Cost = %.2f' % (cost * 10000 / len(train)))
    data_process.visualize(training)

    testing = DataFrame(columns=('Actual Daily', 'Predicted Daily'))
    cost = 0
    for index in range(len(y_test)):
        output = winner_net.activate(x_test[index])
        testing.loc[index] = [y_test[index], output[0]]
        cost += (y_test[index] - output[0]) ** 2
    print('\nTesting Cost = %.2f' % (cost * 10000 / len(test)))
    data_process.visualize(testing)


# no_of_days_ahead_prediction = 1
#
#
# def get_day(date):
#     dt.datetime.strptime(date, '%Y-%m-%d').strftime('%a')


if __name__ == '__main__':
    data = data_process.retrieve()

    print('Preprocessing data...')

    data = data_process.scale(data)
    train, test = data_process.split(data, train_size=.9)  # split into training and testing data
    # x_train = [[train.iloc[index][attribute] for attribute in attributes] +
    #            [1 if get_day(str(train.iloc[index].name)[:-9]) == day else 0 for day in days]
    #            for index in range(len(train) - no_of_days_ahead_prediction)]
    # y_train = [[train.iloc[i + no_of_days_ahead_prediction]['Close']] for i in
    #            range(len(train) - no_of_days_ahead_prediction)]
    # x_test = [[test.iloc[index][attribute] for attribute in attributes] +
    #           [1 if get_day(str(train.iloc[index].name)[:-9]) == day else 0 for day in days]
    #           for index in range(len(test) - no_of_days_ahead_prediction)]
    # y_test = [[test.iloc[i + no_of_days_ahead_prediction]['Close']] for i in
    #           range(len(test) - no_of_days_ahead_prediction)]
    train = data_process.strip(train)
    test = data_process.strip(test)

    # x_train = [[train.iloc[i][attribute] for i in range(index, index + 30) for attribute in attributes]
    #            for index in range(0, len(train) - 1, 30)]
    # y_train = [train.iloc[i]['Close'] for i in range(30, len(train), 30)]
    # x_test = [[test.iloc[i][attribute] for i in range(index, index + 30) for attribute in attributes]
    #           for index in range(0, len(test) - 1, 30)]
    # y_test = [test.iloc[i]['Close'] for i in range(30, len(test), 30)]

    x_train = [[train.iloc[i]['Close'] for i in range(index, index + 30)] for index in range(0, len(train) - 1, 30)]
    y_train = [train.iloc[i]['Close'] for i in range(30, len(train), 30)]
    x_test = [[test.iloc[i]['Close'] for i in range(index, index + 30)] for index in range(0, len(test) - 1, 30)]
    y_test = [test.iloc[i]['Close'] for i in range(30, len(test), 30)]

    # Determine path to configuration file. This path manipulation is here so that the script will run successfully
    # regardless of the current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-evolve_neural_network')
    run(config_path)
