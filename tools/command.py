import argparse


def train_command():
    parser = argparse.ArgumentParser(description='Train a network.')
    parser.add_argument('-s', type=int,
                        help='number of samples per category',
                        dest='n_samples')
    parser.add_argument('-l', type=str,
                        help='transfer layer name',
                        dest='layer_name')
    parser.add_argument('-o', type=argparse.FileType('wb'),
                        help='output file to pickle history to',
                        dest='output_file')
    args = parser.parse_args()
