from dcpg.envs import make_envs

import torch
import numpy as np
import wandb

def render_starting_states(state_buffer):
    num_states = 100
    obs, _ = state_buffer.sample(num_states)
    img = render_obs((obs*255).to(torch.uint8), num_states)

    wandb.log({"state_buffer_examples": wandb.Image(img)})
    

def resize_image(image_matrix, nh, nw):
    image_size = image_matrix.shape
    oh = image_size[0]
    ow = image_size[1]   

    re_image_matrix = np.array([
        np.array([image_matrix[(oh*h // nh)][(ow*w // nw)] for w in range(nw)]) 
        for h in range(nh)
    ])

    return re_image_matrix

def concat_images(image_set, how):
    # dimension of each matrix in image_set
    shape_vals = [imat.shape for imat in image_set]

    # length of dimension of each matrix in image_set
    shape_lens = [len(ishp) for ishp in shape_vals]

    # if all the images in image_set are read in same mode
    channel_flag = True if len(set(shape_lens)) == 1 else False

    if channel_flag:
        ideal_shape = max(shape_vals)
        images_resized = [
            # function call to resize the image
            resize_image(image_matrix=imat, nh=ideal_shape[0], nw=ideal_shape[1]) 
            if imat.shape != ideal_shape else imat for imat in image_set
        ]
    else:
        return False

    images_resized = tuple(images_resized)

    if (how == 'vertical') or (how == 0):
        axis_val = 0
    elif (how == 'horizontal') or (how == 1):
        axis_val = 1
    else:
        axis_val = 1

    # numpy code to concatenate the image matrices
    # concatenation is done based on axis value
    concats = np.concatenate(images_resized, axis=axis_val)
    return concats

def render_obs(obs, n_envs):
    obs = obs.cpu().numpy().transpose((0,2,3,1))
    image_size = int(np.floor(np.sqrt(n_envs)))
    rows = []
    for i in range(image_size):
     row_start_ind = i*image_size
     row = obs[row_start_ind:row_start_ind + image_size]

     rows.append(concat_images(row, how='horizontal'))

    img = concat_images(rows, how='vertical')

    return img