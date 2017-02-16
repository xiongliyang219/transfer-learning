from keras.applications import VGG16
from keras.layers import Dense
from keras.models import Model


def build_transfer_net(output_dim, transfer_layer_name='fc2'):
    """Build a transfer learning CNN based on VGG16."""
    # Start with the VGG16 model trained on the ImageNet dataset.
    base_model = VGG16(weights='imagenet')

    # Attach a softmax layer to the transfer layer as the new output layer.
    transfer_layer = base_model.get_layer(transfer_layer_name)
    output_layer = Dense(output_dim, activation='softmax', name='predictions',init='he_normal')
    predictions = output_layer(transfer_layer.output)
    model = Model(input=base_model.input, output=predictions)

    # Ensure that only the last layer is going to be trained.
    for layer in base_model.layers:
        layer.trainable = False

    return model
