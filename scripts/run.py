import json

from tools.training import transfer_learn


if __name__ == '__main__':
    with open('params.json') as f:
        params = json.load(f)
    for layer_name, nb_sample in params['samples']:
        print('Processing', layer_name, nb_sample)
        output_file = ('../results/urban_tribes-{}-{}.hdf5'
                       .format(layer_name, nb_sample))
        transfer_learn(layer_name, nb_sample, params['nb_epoch'], output_file)
