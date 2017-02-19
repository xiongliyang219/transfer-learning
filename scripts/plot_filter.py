import json
from tools.visualization import plot_filter


if __name__ == '__main__':
    with open('params.json') as f:
        params = json.load(f)
    for image_path, layer_name in params['filters']:
        print('Processing', image_path, layer_name)
        plot_filter(image_path, layer_name, output_dir='../results')
