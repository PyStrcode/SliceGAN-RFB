import os
from torch import nn
import torch
from torch import autograd
import matplotlib.pyplot as plt
import tifffile
import sys
import cv2
import numpy as np
from torch.autograd import Variable


## Training Utils

def load_checkpoint(checkpoint_path, netG):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['Generator_state_dict'])

        print(f"Checkpoint loaded. generate img from epoch {start_epoch}.")
        return start_epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0 

def mkdr(proj,proj_dir,Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """
    pth = proj_dir + '/' + proj
    # print(proj_dir)
    pth_tensorboard = pth + '//tensorboard'
    pth_slice = pth + '//slices'
    pth_checkpoint = pth + '//checkpoint'
    # print(pth_tensorboard
    if Training ==1:
        try:
            os.mkdir(pth)
            os.mkdir(pth_tensorboard)
            os.mkdir(pth_slice)
            os.mkdir(pth_checkpoint)
            return pth + '/'
        except FileExistsError:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/'
            else:
                pth = mkdr(new, proj_dir, Training)
                return pth
        except FileNotFoundError:
            print('The specifified project directory ' + proj_dir + ' does not exist. Please change to a directory that does exist and again')
            sys.exit()
    else:
        return pth + '/' + proj


def weights_init(m):
    """
    Initialises training weights
    :param m: Convolution to be intialised
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def compute_w_div(netD, real_samples, fake_samples):
    k = 2
    p = 6
    real_out = netD(real_samples).view(-1)
    fake_out = netD(fake_samples).view(-1)
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    # real_samples = Variable(real_samples).to(device).requires_grad_(True)
    # real_out = Variable(real_out).to(device).requires_grad_(True)
    # fake_samples = Variable(fake_samples).to(device).requires_grad_(True)
    # fake_out = Variable(fake_out).to(device).requires_grad_(True)
    weight = torch.full((real_samples.size(0),), 1, device=device)
    real_grad = autograd.grad(outputs=real_out,
                              inputs=real_samples,
                              grad_outputs=weight,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1)
    fake_grad = autograd.grad(outputs=fake_out,
                              inputs=fake_samples,
                              grad_outputs=weight,
                              create_graph=True,
                              retain_graph=True,
                              only_inputs=True)[0]
    fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1)
    div_gp = torch.mean(real_grad_norm ** (p / 2) + fake_grad_norm ** (p / 2)) * k / 2
    return div_gp


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda,nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param device:
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty




def calc_eta(steps, time, start, i, epoch, num_epochs):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps:
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch:
    :param num_epochs: totale no. of epochs
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))

## Plotting Utils
def post_proc(img,imtype):
    """
    turns one hot image back into grayscale
    :param img: input image
    :param imtype: image type
    :return: plottable image in the same form as the training data
    """
    try:
        #make sure it's one the cpu and detached from grads for plotting purposes
        img = img.detach().cpu()
    except:
        pass
    if imtype == 'colour':
        return np.int_(255 * (np.swapaxes(img[0], 0, -1)))
    if imtype == 'grayscale':
        return 255*img[0][0]
    else:
        nphase = img.shape[1]
        return 255*torch.argmax(img, 1)/(nphase-1)
        
def test_plotter(img,slcs,imtype,pth, epoch):
    """
    creates a fig with 3*slc subplots showing example slices along the three axes
    :param img: raw input image
    :param slcs: number of slices to take in each dir
    :param imtype: image type
    :param pth: where to save plot
    """
    img = post_proc(img,imtype)[0]
    fig, axs = plt.subplots(slcs, 3)
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :, :], vmin = 0, vmax = 255)
            axs[j, 1].imshow(img[:, j, :, :],  vmin = 0, vmax = 255)
            axs[j, 2].imshow(img[:, :, j, :],  vmin = 0, vmax = 255)
    elif imtype == 'grayscale':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :], cmap = 'gray')
            axs[j, 1].imshow(img[:, j, :], cmap = 'gray')
            axs[j, 2].imshow(img[:, :, j], cmap = 'gray')
    else:
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :])
            axs[j, 1].imshow(img[:, j, :])
            axs[j, 2].imshow(img[:, :, j])
    plt.savefig(pth + f'{epoch}_slices.png')
    plt.close()

def graph_plot(data,labels,pth,name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """

    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def smooth_3d_volume(volume, kernel_size=(3, 3, 3), sigma=1.0):
    smoothed_volume = np.zeros_like(volume)
    for z in range(volume.shape[2]):
        smoothed_slice = cv2.GaussianBlur(volume[:, :, z], kernel_size[:2], sigma)
        smoothed_volume[:, :, z] = smoothed_slice
    return smoothed_volume

def test_img(pth, Project_name, imtype, netG, nz = 64, lf = 4, periodic=False):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where to save image and also where to find the generator
    :param imtype: image type
    :param netG: Loaded generator class
    :param nz: latent z dimension
    :param lf: length factorvim
    :param show:
    :param periodic: list of periodicity in axis 1 through n
    :return:
    """
    # pth = pth + '/' + Project_name
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    netG.cuda()
    noise = torch.randn(1, nz, lf, lf, lf).cuda()
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    with torch.no_grad():
        raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw,imtype)[0]
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]
    tif = np.int_(gb)
    # tif = smooth_3d_volume(np.uint8(tif))
    tifffile.imwrite(pth + 'bn.tif', tif)

    return tif, raw, netG
def load_checkpoint_and_generate_images(checkpoint_pth, Project_name, imtype, netG, nz=64, lf=4, epoch=35, periodic=[0, 1, 1]):
    """
    Load a generator model from a checkpoint and generate synthetic images.

    :param checkpoint_pth: The path to the generator's checkpoint file.
    :param Peoject_name: The path to the generator's checkpoint file.
    :param image_type: Type of the image (not utilized in this function but retained for compatibility).
    :param netG: An instantiated generator model class.
    :param nz: The dimension of the latent space (default: 64).
    :param lf: A factor controlling the size of generated volumes (default: 4).
    :param use_periodicity: Flag indicating whether periodicity should be applied to noise data (default: False).
    :param periodic_axes: Axes along which to apply periodicity (None or list of axes, e.g., [0, 1]).
    :return: Synthetic image data as numpy array, raw output from the generator, and the loaded generator model.
    """
    # Load the generator's state dictionary from the checkpoint
    # print(checkpoint_pth)
    parent_path = os.path.dirname(checkpoint_pth)
    checkpoint_pth_file = parent_path + '/checkpoint/' + '_checkpoint_' + str(epoch) + '.pth'
    print(checkpoint_pth_file)
    load_checkpoint(checkpoint_pth_file, netG)
    netG.eval()
    netG.cuda()

    noise = torch.randn(1, nz, lf, lf, lf).cuda()
    # # Apply periodic boundary conditions if specified
    # if use_periodicity and periodic is not None:
    #     for axis in periodic:
    #         noise[:, :, axis * 2 - 2:axis * 2] = noise[:, :, axis * 2 - 4:axis * 2 - 2]

    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    with torch.no_grad():
        raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw, imtype)[0]
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:, :-1]
        if periodic[2]:
            gb = gb[:, :, :-1]

    tif = np.int_(gb)
    tifffile.imwrite(checkpoint_pth + '_checkpoint_' + str(epoch) + '.tif', tif)
    # The function originally included post-processing steps and saving to TIFF,
    # which are not included here to focus solely on loading the checkpoint and generating data.

    # For consistency with the original function, we could synthesize an output similar to returning processed images,
    # but since there's no actual post_processing function defined, we'll just return raw data.

    return tif, raw, netG


def resize_img(pth, size):
    img = pth 
    src = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
    print(np.unique(src))
    resize_img = cv2.resize(src, size, interpolation=cv2.INTER_NEAREST)
    print(np.unique(resize_img))
    cv2.imwrite(filename="./resizeimg.png", img=resize_img)
    return resize_img


def preprocess_images_isolated_components_mask(image_path, output_paths=None, threshold_value=100, min_areas=[50], erosion_kernel_size=3):

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    original_image = cv2.imread(image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    fig, axs = plt.subplots(len(min_areas) + 1, 3, figsize=(15, 5 * len(min_areas)))
    axs[0, 0].imshow(original_image_rgb)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off') 

    for idx, min_area in enumerate(min_areas):
        _, threshold = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, connectivity=8)
        mask = np.zeros_like(threshold)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                component_mask = (labels == label).astype(np.uint8) * 255
                erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
                eroded_component = cv2.erode(component_mask, erosion_kernel)
                mask += eroded_component

        processed_gray = gray.copy()
        processed_gray[mask == 255] = 0

        axs[idx + 1, 0].imshow(original_image_rgb)
        axs[idx + 1, 0].set_title(f'Processed (Min Area={min_area})')
        axs[idx + 1, 0].axis('off')

        axs[idx + 1, 1].imshow(processed_gray, cmap='gray')
        axs[idx + 1, 1].set_title(f'Processed Grayscale (Min Area={min_area})')
        axs[idx + 1, 1].axis('off')

        axs[idx + 1, 2].imshow(mask, cmap='gray')
        axs[idx + 1, 2].set_title(f'Mask (Min Area={min_area})')
        axs[idx + 1, 2].axis('off')

        if output_paths and idx < len(output_paths):
            cv2.imwrite(output_paths[idx], processed_gray)

    plt.tight_layout()
    plt.show()


def median_filtering_save_and_display(image_path, output_dir, filter_sizes):
    """
    :param image_path:
    :param output_dir:
    :param filter_sizes:
    :return:
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filtered_imgs = []
    for idx, size in enumerate(filter_sizes):
        filtered_img = cv2.medianBlur(img, size)
        filtered_img = filtered_img[:, :, np.newaxis].repeat(3,axis=2)
        
        output_path = os.path.join(output_dir, f'median_filtered_{size}px_{idx}.png')
        
        cv2.imwrite(output_path, filtered_img)
        print(f"Filtered image saved to {output_path}")
        
        unique_pixel_values = np.unique(filtered_img)
        print(f"Pixel value kinds for {size}px kernel: {unique_pixel_values}")

        filtered_imgs.append(filtered_img)

    combined_img = np.hstack(filtered_imgs)
    
    plt.figure(figsize=(15, 5))
    plt.imshow(combined_img)
    plt.title('Median Filtered Images with Different Kernel Sizes')
    plt.axis('off')
    plt.show()
    return filtered_imgs


def detect_and_display_gradients(image_path, threshold_values):
    """
    :param image_path:
    :param threshold_values: list
    :return:
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    num_thresholds = len(threshold_values)
    if num_thresholds < 5:
        cols = 2 
    else:
        cols = 4
    rows = num_thresholds // cols + (num_thresholds % cols > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 5*rows))
    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    for idx, threshold in enumerate(threshold_values):
        row = idx // cols
        col = idx % cols

        mask = cv2.threshold(gradient, threshold, 255, cv2.THRESH_BINARY)[1]

        axs[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[row, col].imshow(mask, cmap='Blues', alpha=0.5) 
        axs[row, col].set_title(f'Threshold: {threshold}')
        axs[row, col].axis('off')

    for idx in range(num_thresholds, rows*cols):
        fig.delaxes(axs.flatten()[idx])

    plt.tight_layout()
    plt.show()


def detect_and_display_canny(image_path, threshold_pairs):
    """
    :param image_path:
    :param threshold_pairs:
    :return:
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    num_thresholds = len(threshold_pairs)
    if num_thresholds < 5:
        cols = 2
    else:
        cols = 4
    rows = num_thresholds // cols + (num_thresholds % cols > 0)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 5 * rows))

    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    for idx, (low_threshold, high_threshold) in enumerate(threshold_pairs):
        row = idx // cols
        col = idx % cols

        canny_edges = cv2.Canny(gray, low_threshold, high_threshold)

        axs[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[row, col].imshow(canny_edges, cmap='Blues', alpha=0.8)
        axs[row, col].set_title(f'Canny Edges: Low={low_threshold}, High={high_threshold}')
        axs[row, col].axis('off')

    for idx in range(num_thresholds, rows * cols):
        fig.delaxes(axs.flatten()[idx])

    plt.tight_layout()
    plt.show()


def detect_and_display_gradients_with_feature_check(sub_image: object, threshold_values: object, edge_threshold: object = 0.1) -> object:

    img = cv2.imread(sub_image, cv2.IMREAD_GRAYSCALE)

    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    num_thresholds = len(threshold_values)
    cols, rows = (2, num_thresholds // 2 + num_thresholds % 2) if num_thresholds < 5 else (4, num_thresholds // 4 + num_thresholds % 4)

    fig, axs = plt.subplots(rows, cols, figsize=(10, 5*rows))

    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    overall_has_features = False
    for idx, threshold in enumerate(threshold_values):
        row = idx // cols
        col = idx % cols

        mask = (gradient > threshold).astype(np.uint8) * 255

        edge_count = np.sum(mask)
        total_pixels = np.prod(mask.shape)
        has_features = edge_count / total_pixels > edge_threshold
        overall_has_features |= has_features

        axs[row, col].imshow(img, cmap='gray')
        axs[row, col].imshow(mask, cmap='Blues', alpha=0.5)
        title_text = f'Gradient: Threshold={threshold}\nFeatures: {"Yes" if has_features else "No"}'
        axs[row, col].set_title(title_text)
        axs[row, col].axis('off')

    for idx in range(num_thresholds, rows*cols):
        fig.delaxes(axs.flatten()[idx])

    plt.tight_layout()
    plt.show()
    return overall_has_features

def crop_3d_volume(volume, start_indices, end_indices):

    assert all(s >= 0 and e <= dim for s, e, dim in zip(start_indices, end_indices, volume.shape[2:]))

    cropped_volume = volume[:, :,
                     start_indices[0]:end_indices[0],
                     start_indices[1]:end_indices[1],
                     start_indices[2]:end_indices[2]]

    return cropped_volume



