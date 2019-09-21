import numpy as np
import os
import umap

from bokeh import plotting as bk
from bokeh.io import output_notebook, show, push_notebook
from bokeh.layouts import gridplot, row, widgetbox
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, Slider
from bokeh.plotting import figure
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import (manifold, decomposition,
        ensemble, discriminant_analysis, random_projection)
from sklearn.datasets import load_digits
from sklearn.decomposition import NMF, PCA, KernelPCA

imgs_path = '/tmp/umap-mnist-imgs'

def write_pngs(d):

    if not os.path.exists(imgs_path):
        os.makedirs(imgs_path)

    img_paths = []
    for i, pxls in enumerate(d):
        img_path = f'{imgs_path}/{i:04}.png'
        if not os.path.exists(img_path):
            img = Image.new('L', (8, 8))
            img.putdata([ int(255 - p*255) for p in pxls ])
            img.save(img_path)
        img_paths.append(img_path)

    return img_paths

def labels_to_colors(labels):
    l_min = float(min(labels))
    l_max = float(max(labels))
    colors = []
    for l in labels:
        ratio = 2 * (l - l_min) / (l_max - l_min)
        b = int(max(0, 255*(1 - ratio)))
        r = int(max(0, 255*(ratio - 1)))
        g = 255 - b - r
        colors.append(f'#{r:02x}{g:02x}{b:02x}')
    return colors

def make_plot(e, labels, imgs):

    colors = labels_to_colors(labels)

    source = ColumnDataSource(data={
        'x': e[:,0],
        'y': e[:,1],
        'value': labels,
        'fill_color': colors,
        'imgs': imgs })

    hover = HoverTool(tooltips='''
        <div>
            <div>
                <img src="@imgs" height="16" alt="@imgs" width="16" border="2px"
                        style="float: left; margin: 0 15px 15px 0;" />
                <span style="float: right; font-size: 16px; font-weight: bold;">
                    @value
                </span>
            </div>
        </div>''')

    plot = figure(tools=[hover, 'reset,wheel_zoom,pan'])
    plot.circle('x', 'y', size=8, source=source, line_color=None, fill_color='fill_color')

    return plot

def get_umap_embedding(d, **kwargs):
    return umap.UMAP(**kwargs).fit_transform(d)

def get_pca_embedding(d, **kwargs):
    return PCA(**kwargs).fit_transform(d)

def get_t_sne_embedding(d, **kwargs):
    return manifold.TSNE(**kwargs).fit_transform(d)

def get_grid(embedding_funcs, d, labels, imgs, **kwargs):
    return gridplot([[ make_plot(embed(d, **kwargs), labels, imgs) for embed in embedding_funcs ]])

if __name__ == '__main__':

    digits = load_digits()
    labels = digits.target

    img_paths = write_pngs(digits.data)

    #e = get_umap_embedding(digits.data, n_components=2)
    #e = get_pca_embedding(digits.data, n_components=2)
    #e = get_t_sne_embedding(digits.data, n_components=2)
    #show(make_plot(e, labels, img_paths))

    #embeddings =  [
            #get_umap_embedding,
            #get_pca_embedding,
            #get_t_sne_embedding ]

    n_neighbors = [ 2, 5, 10, 15, 20, 50, 100, 200, 500 ]
    embeddings = [ lambda d, **kwargs: get_umap_embedding(d, n_neighbors=n, **kwargs) for n in n_neighbors ]
    grid = get_grid(embeddings, digits.data, labels, img_paths, n_components=2)

    show(grid)

