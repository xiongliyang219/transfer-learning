from pathlib import Path

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import numpy as np


def load_data(path='data/pictures_all', validation_split=0.1, test_split=0.1,
              seed=233, images_per_category=None):
    # Gather and shuffle image paths.
    repo_path = Path(__file__).resolve().parents[2]
    image_paths = list((repo_path / path).glob('*.jpg'))
    np.random.seed(seed)
    np.random.shuffle(image_paths)

    # Gather categories and sort them in alphabet order.
    categories = list(set([get_category(path) for path in image_paths]))
    categories.sort()

    # Split paths into train, val and test sets.
    paths_train = []
    paths_val = []
    paths_test = []
    for cat in categories:
        paths_cat = [path for path in image_paths if get_category(path) == cat]
        vt_edge = int(len(paths_cat) * validation_split)
        tt_edge = int(len(paths_cat) * (validation_split + test_split))
        paths_val += paths_cat[:vt_edge]
        paths_test += paths_cat[vt_edge:tt_edge]
        if images_per_category:
            paths_train += paths_cat[tt_edge:tt_edge + images_per_category]
        else:
            paths_train += paths_cat[tt_edge:]

    # Load images and labels from paths.
    x_train, y_train = load_images(paths_train, categories)
    x_val, y_val = load_images(paths_val, categories)
    x_test, y_test = load_images(paths_test, categories)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_category(image_path):
    return image_path.name.split('_')[0]


def load_images(paths, categories):
    xs = []
    ys = []
    for path in paths:
        x = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(x)
        xs += [x]
        ys += [categories.index(get_category(path))]
    xs = preprocess_input(np.stack(xs))
    ys = to_categorical(ys, nb_classes=len(categories))
    return xs, ys
