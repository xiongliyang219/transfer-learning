from pathlib import Path

from keras.applications import VGG16
from keras.models import Model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

from tools.datasets.urban_tribes import load_images


def plot_filter(image_path, layer_name, output_dir):
    base_model = VGG16(weights='imagenet')
    x = load_images([image_path])
    model = Model(input=base_model.input,
                  output=base_model.get_layer(layer_name).output)
    layer_output = model.predict(x)

    fig = plt.figure(figsize=(12, 12))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(8, 8),
                    axes_pad=0.0,
                    share_all=True,
                    label_mode="L",
                    cbar_location="top",
                    cbar_mode="single")

    for i in range(64):
        im = grid[i].imshow(layer_output[0, :, :, i], interpolation="nearest")
    grid.cbar_axes[0].colorbar(im)

    for cax in grid.cbar_axes:
        cax.toggle_label(False)

    for ax in grid.axes_all:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    output_dir = Path(output_dir)
    fig_file = '{}-{}.pdf'.format(Path(image_path).stem, layer_name)
    plt.savefig(str(output_dir / fig_file))
