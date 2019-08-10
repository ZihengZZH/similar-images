import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def show_similar_image(id, im, idx, img):
    """
    para id: target ID
    para im: target image
    """
    if type(idx) != np.ndarray:
        idx = np.array([idx])
    fig = plt.figure(figsize=(8,8))
    w = (len(idx)+1)//2
    gs = gridspec.GridSpec(2, w)
    for i in range(0,len(idx)+1):
        if i == 0:
            ax = fig.add_subplot(gs[0,i])
            ax.imshow(im[id[0],:,:,0])
            ax.title.set_text('Target (test)')
        else:
            ax = fig.add_subplot(gs[0,i]) if i < w else fig.add_subplot(gs[1,i-w])
            ax.imshow(img[idx[i-1],:,:,0])
            ax.title.set_text('similar (train) #%d' % (i))
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('images/inference.png')