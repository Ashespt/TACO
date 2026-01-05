import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from PIL import Image
from scipy.ndimage import zoom
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
import pandas as pd
umap = UMAP(n_components=2)
tsne = TSNE(n_components=2, random_state=42)
pca = PCA(n_components=2)
pca_1d = PCA(n_components=1)
def tsne_visual(features_path,save_path):
    features_labels = np.load(features_path)
    features = features_labels['feature']
    labels = features_labels['label']
    data_2d = tsne.fit_transform(features)
    colors = ['r', 'b']
    label_names = ['Siemens Vision Quadra','uExplorer']
    for i, color in enumerate(colors):
        subset = data_2d[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i], alpha=0.5)
    plt.legend()
    # scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter)
    plt.title('t-SNE visualization of validation dataset')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.savefig(save_path)
    plt.close()

def umap_visual_fe(features,labels,save_path):
    unique, counts = np.unique(labels, return_counts=True)
    most_frequent_value = unique[np.argmax(counts)]
    sorted_indices = np.argsort(counts)[::-1]
    sorted_values = unique[sorted_indices]
    data_2d = umap.fit_transform(features[np.isin(labels, sorted_values[1:5])])
    umap_df = pd.DataFrame(data_2d, columns=["UMAP1", "UMAP2"])
    umap_df['label'] = labels[np.isin(labels, sorted_values[1:5])]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="label", palette="Set1", data=umap_df, s=60, edgecolor=None)
    plt.title("UMAP projection of the feature data", fontsize=16)
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    plt.savefig(save_path)
    plt.close()

def pca_visual(features_path,save_path):
    features_labels = np.load(features_path)
    features = features_labels['feature']
    labels = features_labels['label']
    data_2d = pca.fit_transform(features)
    colors = ['r', 'b']
    label_names = ['Siemens Vision Quadra','uExplorer']
    for i, color in enumerate(colors):
        subset = data_2d[labels == i]
        plt.scatter(subset[:, 0], subset[:, 1], c=color, label=label_names[i],alpha=0.5)
    # scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter)
    plt.legend()
    plt.title('PCA visualization of validation dataset')
    plt.xlabel('PCA feature 1')
    plt.ylabel('PCA feature 2')
    plt.savefig(save_path)
    plt.close()

import matplotlib as mpl
from matplotlib import cm
import nibabel as nib
cmap_name = 'cividis'
# cmap_name = 'Pastel1'
def pca_1d_visual(feature1,features,save_path,dim=192,label=None):
    _,h,w,d,dim = feature1.shape
    if label is not None:
        _,_,h_l,w_l,d_l = label.shape
        zoom_factors = (h / h_l, w / w_l, d / d_l)
        downsampled_mask = zoom(label[0,0,...], zoom_factors, order=0)
        downsampled_mask = downsampled_mask.astype(int)
        data = {'features': feature1[0][downsampled_mask>0], 'labels': downsampled_mask[downsampled_mask>0]}
        np.save(save_path.replace('.nii.gz','.npy'),data)
    # import pdb;pdb.set_trace()
    # umap_visual_fe(feature1[0][downsampled_mask>0],downsampled_mask[downsampled_mask>0],save_path.replace('.nii.gz','_umap.png'))
    
    features = features.reshape(-1,dim)
    t = features.shape[0]
    t = int(np.ceil(math.pow(t/16, 1/3)))
    data_1d = pca_1d.fit_transform(features)
    
    data_1d = data_1d.reshape(t*4,t*4,-1)
    d = data_1d.shape[-1]
    
    # new_image = nib.Nifti1Image(data_1d, np.eye(4)) 
    # new_image.set_data_dtype(np.float32) 
    
    # nib.save(new_image, save_path)  
    #slice
    data = data_1d[:,:,int(d//2)]
    plt.imshow(data, cmap=cmap_name)
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize()
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = scalarMap.to_rgba(data)

    image = Image.fromarray((colors[:, :, :3]*256).astype(np.uint8))
    image.save(save_path.replace('.nii.gz','.png'))
    plt.close()

