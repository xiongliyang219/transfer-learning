import json

import matplotlib.pyplot as plt

from tools.training import Session

if __name__ == '__main__':
    with open('params.json') as f:
        params = json.load(f)
    nb_samples = [2, 4, 8, 16]
    accuracy = []
    layer_name = 'fc2'

    # Plot loss and accuracy vs iterations
    for metric, name in [('loss', 'loss'), ('acc', 'accuracy')]:
        plt.figure()
        for nb_sample in nb_samples:
            print('Processing', layer_name, nb_sample, metric)
            output_file = '../results/urban_tribes-{}-{}.hdf5'.format(
                layer_name, nb_sample)
            history = Session.load_history(output_file)
            if metric is 'acc':
                accuracy += [history['val_acc'][-1]]
            p = plt.plot(history[metric], '--',
                         label='{} train'.format(nb_sample))
            plt.plot(history['val_' + metric], c=p[-1].get_color(),
                     label='{} test'.format(nb_sample))
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel(name)
        fig_path = ('../results/urban_tribes-{}-{}.pdf'
                    .format(layer_name, metric))
        plt.savefig(fig_path)

    # Plot classification accuracy vs. number of samples used per class
    print(nb_samples)
    print(accuracy)
    plt.figure()
    plt.plot(nb_samples, accuracy, 'o-')
    plt.xlabel('number of samples per category')
    plt.ylabel('accuracy')
    fig_path = '../results/urban_tribes-fc2-acc-sample.pdf'
    plt.savefig(fig_path)

    # Plot intermediate convolutional layers
    nb_sample = 8
    for metric, name in [('loss', 'loss'), ('acc', 'accuracy')]:
        plt.figure()
        for layer_name in ['block1_pool', 'block2_pool', 'block3_pool',
                           'block4_pool', 'block5_pool', 'fc2']:
            print('Processing', layer_name, nb_sample, metric)
            output_file = '../results/urban_tribes-{}-{}.hdf5'.format(
                layer_name, nb_sample)
            history = Session.load_history(output_file)
            p = plt.plot(history[metric][:params['nb_epoch']], '--',
                         label='{} train'.format(layer_name))
            plt.plot(history['val_' + metric][:params['nb_epoch']],
                     c=p[-1].get_color(),
                     label='{} test'.format(layer_name))
            plt.legend(loc='best')
            plt.xlabel('epoch')
            plt.ylabel(name)
        fig_path = ('../results/urban_tribes-{}-{}.pdf'
                    .format('inter_8_per_cat', metric))
        plt.savefig(fig_path)
