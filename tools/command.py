import argparse

from tools.training import transfer_learn


def transfer_learn_command():
    parser = argparse.ArgumentParser(
        description='Transfer learning for image classification.'
    )
    parser.add_argument('-e', '--epoch', type=int,
                        help='number of epochs to train',
                        dest='nb_epoch')
    parser.add_argument('-l', '--layer', type=str,
                        help='transfer layer name',
                        dest='layer_name')
    parser.add_argument('-o', '--output', type=str,
                        help='name of the output file to pickle history to',
                        dest='output_file')
    parser.add_argument('-s', '--sample', type=int,
                        help='number of samples per category',
                        dest='nb_sample')
    args = parser.parse_args()

    transfer_learn(args.layer_name, args.nb_sample, args.nb_epoch,
                   args.output_file)
