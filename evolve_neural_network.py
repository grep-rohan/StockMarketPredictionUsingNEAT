"""Evolve neural network for stock market prediction."""
import datetime as dt
import time

from pandas import DataFrame

import preprocess

start = time.clock()
import os
from pastalog import Log

import neat

import visualize

reporter = neat.StdOutReporter(True)
stats = neat.StatisticsReporter()

log = Log('http://localhost:8120', 'A')

train = test = data = inputs = outputs = test_inputs = test_outputs = None

columns = ('Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close', 'Exchange')


# noinspection PyBroadException
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        cost1 = cost2 = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for input, output in zip(inputs, outputs):
            predicted_output = net.activate(input)
            cost1 += (predicted_output[0] - output[0]) ** 2
            cost2 += (predicted_output[1] - output[1]) ** 2
        cost1 /= 2
        cost2 /= 2
        genome.fitness = (-cost1 - cost2)
    try:
        # pass
        if reporter.generation > 0:
            log.post('bestFitness', value=stats.best_genome().fitness, step=reporter.generation - 1)
            log.post('averageFitness', value=stats.get_fitness_mean()[reporter.generation - 1],
                     step=reporter.generation - 1)
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
    winner = p.run(eval_genomes, 600)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for input, output in zip(inputs, outputs):
    #     output = winner_net.activate(input)
    #     print("input {!r}, expected output {!r}, got {!r}".format(input, preprocess.descale(output, data),
    #                                                               preprocess.descale(output, data)))

    node_names = {-1: columns[0], -2: columns[1], -3: columns[2], -4: columns[3], -5: columns[4], -6: columns[5],
                  -7: columns[6], 0: 'Predicted Daily', 1: 'Predicted Weekly'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    daily = DataFrame(columns=('Actual Daily', 'Predicted Daily'))
    weekly = DataFrame(columns=('Actual Weekly', 'Predicted Weekly'))
    count = 0
    # print('\n\n\nTesting...\n')
    cost1 = 0
    cost2 = 0
    for ip, op in zip(test_inputs, test_outputs):
        output = winner_net.activate(ip)
        daily.loc[count] = [op[0], output[0]]
        weekly.loc[count] = [op[1], output[1]]
        count += 1
        cost1 += (output[0] - op[0]) ** 2
        cost2 += (output[1] - op[1]) ** 2
        # print("expected output {!r}, got {!r}".format(op, output))
    print('Test Cost =', -(cost1 + cost2) / 2)
    preprocess.visualize(daily)
    preprocess.visualize(weekly)


if __name__ == '__main__':
    data = preprocess.retrieve()
    data_scaled = preprocess.scale(data)
    train, test = preprocess.split(data_scaled, train_percent=.9)
    inputs = [[train.iloc[i][j] for j in columns] +
              [1 if dt.datetime.strptime(str(train.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Mon' else 0,
               1 if dt.datetime.strptime(str(train.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Tue' else 0,
               1 if dt.datetime.strptime(str(train.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Wed' else 0,
               1 if dt.datetime.strptime(str(train.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Thu' else 0,
               1 if dt.datetime.strptime(str(train.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Fri' else 0]
              for i in range(len(train) - 7)]
    outputs = [[train.iloc[i + j]['Close'] for j in [1, 7]] for i in range(len(train) - 7)]
    test_inputs = [[test.iloc[i][j] for j in columns] +
                   [1 if dt.datetime.strptime(str(test.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Mon' else 0,
                    1 if dt.datetime.strptime(str(test.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Tue' else 0,
                    1 if dt.datetime.strptime(str(test.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Wed' else 0,
                    1 if dt.datetime.strptime(str(test.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Thu' else 0,
                    1 if dt.datetime.strptime(str(test.iloc[i].name)[:-9], '%Y-%m-%d').strftime('%a') == 'Fri' else 0]
                   for i in range(len(test) - 7)]
    test_outputs = [[test.iloc[i + j]['Close'] for j in [1, 7]] for i in range(len(test) - 7)]
    # Determine path to configuration file. This path manipulation is here so that the script will run successfully
    # regardless of the current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-evolve_neural_network')
    run(config_path)

print('\n\nTotal Time Taken: ', time.clock() - start)
