import pickle

from tools import build_transfer_net
from tools.datasets.urban_tribes import load_data


def transfer_learn(layer_name, nb_sample, nb_epoch, output_file):
    """Transfer learning for image classification.

    Args:
        layer_name: Transfer layer name.
        nb_sample: Number of samples per categories.
        nb_epoch: Number of epochs to train.
        output_file: Name of the output file to pick history to.
    """
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
        load_data(images_per_category=nb_sample)
    model = build_transfer_net(output_dim=11,
                               transfer_layer_name=layer_name)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=nb_sample,
                        nb_epoch=nb_epoch, validation_data=(x_val, y_val))
    with open(output_file, 'wb') as f:
        pickle.dump(history.history, f)
