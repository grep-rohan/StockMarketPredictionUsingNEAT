"""Evolve neural network for stock market prediction."""
import datetime as dt
import random
import string
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
attributes = ('Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted Close', 'Exchange')
days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri')
node_names = {-1: attributes[0], -2: attributes[1], -3: attributes[2], -4: attributes[3], -5: attributes[4],
              -6: attributes[5],
              -7: attributes[6], -8: days[0], -9: days[1], -10: days[2], -11: days[3], -12: days[4],
              0: 'Predicted Close'}
random_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
pastslog = Log('http://localhost:8120', random_id)

train = test = data = inputs = outputs = test_inputs = test_outputs = None


# noinspection PyBroadException
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        cost = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for input, output in zip(inputs, outputs):
            predicted_output = net.activate(input)
            cost += (predicted_output[0] - output[0]) ** 2
        genome.fitness = -.5 * cost
    try:
        # pass
        gen = reporter.generation - 1
        if gen > 0:
            visualize.draw_net(config=config, genome=stats.best_genome(), view=False, node_names=node_names)
            pastslog.post('best', value=stats.best_genome().fitness, step=gen)
            # pastslog.post('+1 stdev', value=stats.get_fitness_mean()[gen] + stats.get_fitness_stdev()[gen], step=gen)
            pastslog.post('avg', value=stats.get_fitness_mean()[gen], step=gen)
            # pastslog.post('-1 stdev', value=stats.get_fitness_mean()[gen] - stats.get_fitness_stdev()[gen], step=gen)
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
    winner = p.run(eval_genomes, 5)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    # print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # for input, output in zip(inputs, outputs):
    #     output = winner_net.activate(input)
    #     print("input {!r}, expected output {!r}, got {!r}".format(input, preprocess.descale(output, data),
    #                                                               preprocess.descale(output, data)))

    visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    daily = DataFrame(columns=('Actual Daily', 'Predicted Daily'))
    count = 0
    print('\n\n\nTesting...\n')
    cost = 0
    for ip, op in zip(test_inputs, test_outputs):
        output = winner_net.activate(ip)
        daily.loc[count] = [op[0], output[0]]
        count += 1
        cost += (output[0] - op[0]) ** 2
        # print("expected output {!r}, got {!r}".format(op, output))
    print('Test Cost =', -.5 * cost)
    preprocess.visualize(daily)


no_of_days_ahead_prediction = 1

if __name__ == '__main__':
    data = preprocess.retrieve()
    train, test = preprocess.split(data, train_percent=.6)
    train = preprocess.scale(train)
    test = preprocess.scale(test)
    inputs = [[train.iloc[index][attribute] for attribute in attributes] +
              [1 if dt.datetime.strptime(str(train.iloc[index].name)[:-9], '%Y-%m-%d').strftime('%a') == day else 0
               for day in days]
              for index in range(len(train) - no_of_days_ahead_prediction)]
    outputs = [[train.iloc[i + no_of_days_ahead_prediction]['Close']] for i in
               range(len(train) - no_of_days_ahead_prediction)]
    test_inputs = [[test.iloc[index][attribute] for attribute in attributes] +
                   [1 if dt.datetime.strptime(str(test.iloc[index].name)[:-9], '%Y-%m-%d').strftime('%a') == day else 0
                    for day in days]
                   for index in range(len(test) - no_of_days_ahead_prediction)]
    test_outputs = [[test.iloc[i + no_of_days_ahead_prediction]['Close']] for i in
                    range(len(test) - no_of_days_ahead_prediction)]

    # Determine path to configuration file. This path manipulation is here so that the script will run successfully
    # regardless of the current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-evolve_neural_network')
    run(config_path)

print('\n\nTotal Time Taken: ', time.clock() - start)
