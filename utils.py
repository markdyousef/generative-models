import matplotlib.pyplot as plt

def plot_multi(images, dim=(4,4), figsize=(6,6), **kwargs):
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(*((dim) + (i+1,)))
        plt.imshow(img, **kwargs)
        plt.axis('off')
    plt.tight_layout()
