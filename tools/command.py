import argparse


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
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'),
                        help='output file to pickle history to',
                        dest='output_file')
    parser.add_argument('-s', '--sample', type=int,
                        help='number of samples per category',
                        dest='nb_sample')
    args = parser.parse_args()
