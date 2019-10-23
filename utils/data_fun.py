"""
Functions about data loading.
"""
import cv2
import os
import numpy as np


def norm_data(arr_data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Given array of data, and image mean & std, normalize the data array.
    :param arr_data: np array, shape must be  [n_color_channels, n_frames, image_size, image_size] .e.g. np.array((3, 79, 224, 224))
    :param mean: a list of RGB value mean
    :param std: a list of RGB value std
    :return: normlized data array
    """
    arr_data /= 255.0
    arr_data = np.transpose(arr_data, (1, 2, 3, 0))
    arr_data = (arr_data - mean) / std
    arr_data = np.transpose(arr_data, (3, 0, 1, 2))

    return arr_data




def load_frames_from_folder_as_npy(folder_frames, img_size=224, suffix='.jpg'):
    """
    Load frames from a folder which contains frames from one video or clip.
    Image names should be like '00000.jpg, 00001.jpg..'
    :param folder_frames: folder path of the frames.
    :param img_size: output single frame size
    :param suffix: frame images' file suffix.
    :return: numpy array: [n_color_channels, n_frames, image_size, image_size] .e.g. np.array((3, 79, 224, 224))
    """
    assert os.path.isdir(folder_frames), "folder_frames doesn't exist."
    l_frame = list()
    for img_name in os.listdir(folder_frames):
        if img_name.endswith(suffix):
            file_img = os.path.join(folder_frames, img_name)
            l_frame.append(file_img)

    # sort
    l_frame.sort()

    num_frames = len(l_frame)
    out_arr = np.zeros((3, num_frames, img_size, img_size))

    for i in range(num_frames):
        frame = l_frame[i]
        arr_img = cv2.imread(frame)
        arr_img_resize = cv2.resize(arr_img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        arr_img_final = np.transpose(arr_img_resize, (2, 0, 1))

        out_arr[:, i, :, :] = arr_img_final

    return norm_data(out_arr).astype(np.float32)


