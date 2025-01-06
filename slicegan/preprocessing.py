import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
import cv2

def batch(data,type,l, sf):
    """
    Generate a batch of images randomly sampled from a training microstructure
    :param data: data path
    :param type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    Testing = False
    # l = 300
    if type in ['png', 'jpg', 'tif2D']:
        datasetxyz = []
        for img in data:
            img = plt.imread(img) if type != 'tif2D' else tifffile.imread(img)
            print(img.shape, np.unique(img))
            if len(img.shape)>2:
                img = img[:, :, 0]
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            phases = np.unique(img)
            data = np.empty([32 * 90, len(phases), l, l], dtype=np.float32)
            for i in range(32 * 90):
                found_feature = False
                while not found_feature:
                    print(img.shape)
                    x = np.random.randint(1, x_max - l-1)
                    y = np.random.randint(1, y_max - l-1)
                    sub_img = img[ x:x+l, y:y+l]
                    print(sub_img.shape)
                    if check_feature(sub_img, threshold=3, low_edge_threshold=0, high_edge_threshold=1000, test=0):
                        found_feature = True
                        for cnt, phs in enumerate(phases):
                            img1 = np.zeros([l, l], dtype=np.float32)
                            img1[img[x:x + l, y:y + l] == phs] = 1
                            data[i, cnt, :, :] = img1
                    else:
                        continue

            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :]+2*data[j, 1, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='tif3D':
        datasetxyz=[]
        img = np.array(tifffile.imread(data[0]))
        img = img[::sf,::sf,::sf]
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)
        for dim in range(3):
            data = np.empty([32 * 9, len(vals), l, l])
            print('dataset ', dim)
            for i in range(32*9):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max - l)
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([l,l])
                    if dim==0:
                        img1[img[lay, y:y + l, z:z + l] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + l,lay, z:z + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l,lay] == phs] = 1
                    data[i, cnt, :, :] = img1[:,:]

            if Testing:
                for j in range(2):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(1)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='colour':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            img = img[::sf,::sf,:]
            ep_sz = 32 * 900
            data = np.empty([ep_sz, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + l, y:y + l,0]
                data[i, 1, :, :] = img[x:x + l, y:y + l,1]
                data[i, 2, :, :] = img[x:x + l, y:y + l,2]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='grayscale':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = np.empty([32 * 900, 1, l, l])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                subim = img[x:x + l, y:y + l]
                data[i, 0, :, :] = subim
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
            
    return datasetxyz

def check_feature(sub_image, threshold = 2, low_edge_threshold = 20, high_edge_threshold=80, test = 1):

    img_cv = (sub_image * 255).astype(np.uint8)
    img = np.expand_dims(img_cv, axis=-1)
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    gradient = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    mask = (gradient > threshold).astype(np.uint8) * 255

    edge_count = np.sum(mask)
    total_pixels = np.prod(mask.shape)
    has_features = (edge_count / total_pixels > low_edge_threshold) and (edge_count / total_pixels < high_edge_threshold)

    if test:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        axs[1].imshow(img, cmap='gray')
        axs[1].imshow(mask, cmap='Blues', alpha=0.5)
        title_text = f'Gradient: Threshold={threshold} edgecount={edge_count / total_pixels} \nFeatures: {"Yes" if has_features else "No"} '
        axs[1].set_title(title_text)
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return has_features
