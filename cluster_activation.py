import json
import os

import numpy as np
from PIL import Image
from data import ImageDataset, pixel_to_continuous_coordinate, positional_encoding
from parse import parser
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import utils

def get_activations(net_act):
    def hook(model, input, output):
        net_act.append(output.detach().cpu().numpy())
    return hook

def get_coords(config):
    img = Image.open(config.imagepath)
    img = np.array(img)
    if config.drop_alpha:
        img = img[:, :, :-1]
    height = img.shape[0]
    width = img.shape[1]
    coords = []
    for i in range(height):
        for j in range(width):
            if config.use_posenc:
                coords.append(positional_encoding(pixel_to_continuous_coordinate(img.shape[:2], i, j), config))
            else:
                coords.append(pixel_to_continuous_coordinate(img.shape[:2], i, j))
    return coords


if __name__ == "__main__":
    config = parser.parse_args()
    model, loader, optimizer, palette_optimizer = utils.setup(config)
    layer_id = 2
    img_object = Image.open(config.imagepath)
    img = np.array(img_object)
    img_object.putalpha(128)
    model.eval()
    activations = []

    model.base.layers[layer_id].register_forward_hook(get_activations(activations))
    if not os.path.exists(os.path.join(config.expdir, "kmeans")):
        os.mkdir(os.path.join(config.expdir, "kmeans"))

    coords = get_coords(config)

    for i in range(0, len(coords), config.batchsize):
        batch = torch.tensor(coords[i:min(i + config.batchsize, len(coords))], dtype=torch.float32).to(config.device)
        model(batch)
    activations = np.array(activations).reshape(-1, config.hidden_size)

    k_means = MiniBatchKMeans(n_clusters=16, batch_size=1024, max_iter=1000, verbose=0, n_init=20).fit(activations)

    # centers = k_means.cluster_centers_
    results = k_means.predict(activations)

    x = np.repeat(np.expand_dims(np.array(range(img.shape[1])), axis=0), img.shape[0], axis=0).flatten()
    y = np.repeat(np.expand_dims(np.array(range(img.shape[0])), axis=0), img.shape[1], axis=0).flatten('F')
    plt.scatter(x, y, c=results.tolist(), cmap='tab20')

    ax = plt.subplot(111)
    imagebox = OffsetImage(img_object)
    ax.add_artist(imagebox)

    plt.show()

    # pca = PCA(n_components=2)
    # pca = pca.fit(activations)
    # x_result = np.array(pca.transform(activations)).T
    # plt.scatter(x=x_result[0], y=x_result[1], c=results.tolist(), cmap='tab20')
    # plt.show()
