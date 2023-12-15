import uuid
import numpy as np
import scipy.special
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import imageio


# Plotting
def save_fig(fig, path, dpi=100, tight_layout_kwargs={}):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(**tight_layout_kwargs)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(path)
    print("Saved to "+ str(path))
    plt.close(fig)


def make_gif(img_paths, gif_path, fps):
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    images = []
    for img_path in tqdm(img_paths):
        images.append(imageio.imread(img_path))
    imageio.mimsave(gif_path, images, duration=1 / fps)
    print("Saved to " + str(gif_path))


def one_hot_to_int(one_hot):
    """Convert a one-hot vector to an int
    Args
        one_hot [dim]

    Returns (int)
    """
    assert len(one_hot.shape) == 1
    return (one_hot * torch.arange(len(one_hot), device=one_hot.device)).sum().item()


def get_one_ids(x):
    """Get a list of ints for which x[i] is one or greater (goal block)
    Args
        x [dim]

    Returns (list of ints)
    """
    return list(torch.where(x >= 1)[0].cpu().numpy())


def sample_ancestral_index(log_weight):
    """Sample ancestral index using systematic resampling for SMC.

    Args
        log_weight: log of unnormalized weights, tensor
            [num_weights] or [batch_size, num_weights]
    Returns
        zero-indexed ancestral index: LongTensor [num_weights] or [batch_size, num_weights]
    """
    if len(log_weight.shape) == 1:
        return sample_ancestral_index(log_weight[None])[0]

    batch_size, num_weights = log_weight.shape
    indices = np.zeros([batch_size, num_weights])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_weights)) / num_weights

    normalized_weights = scipy.special.softmax(log_weight, axis=1)

    # np.ndarray [batch_size, num_weights]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # hack to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(cumulative_weights, axis=1, keepdims=True)

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    return indices.astype(int)


def get_tmp_dir():
    temp_dir = "tmp" + str(str(uuid.uuid4())[:8])
    return temp_dir
