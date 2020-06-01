from skimage import io
from skimage.transform import resize
import glob

import ascent
import numpy as np


def get_image(path_to_image, image_size):
    image = io.imread(path_to_image)
    if image.ndim == 3:
        image = resize(image, (image_size, image_size, image.shape[-1]))
    elif image.ndim == 2:
        image = resize(image, (image_size, image_size))
    return image


def get_rect_image(path_to_image, height, width):
    image = io.imread(path_to_image)
    if image.ndim == 3:
        image = resize(image, (height, width, image.shape[-1]))
    elif image.ndim == 2:
        image = resize(image, (height, width))
    return image


def load_ascent_images(folder_path, image_size, image_num=None, offset=0):
    path_images = glob.glob(folder_path + r"/*.jpg")
    if image_num:
        if offset + image_num > len(path_images):
            path_images = path_images[offset:len(path_images)]
        else:
            path_images = path_images[offset:offset + image_num]
    images = []
    for path in path_images:
        images.append(get_image(path, image_size))
    images = np.array(images)

    # for multichannel images process each channel separately
    if images.ndim == 4:
        ascent_images = np.empty((2 * image_size, image_size, image_size, images.shape[-1]))
        for ch in range(images.shape[-1]):
            chIm = images[:, :, :, ch].reshape((images.shape[0], image_size, image_size))
            ascent_images[:, :, :, ch] = ascent.generate_ascent(chIm)
    # for one channel images use standart function
    if images.ndim == 3:
        ascent_images = ascent.generate_ascent(images)

    return ascent_images


def load_inner_ascent_images(folder_path, image_size, image_num=None):
    path_images = glob.glob(folder_path + r"/*.jpg")
    if image_num:
        path_images = path_images[:image_num]

    images = []
    for num, path in enumerate(path_images):
        try:
            # print(num)
            # В контескте одного 2D изображения - сэмпл это строка (первый шейп
            image = get_rect_image(path, image_size + 1, image_size)
            if image.ndim == 3:
                ascent_image = np.empty((image.shape[1], image.shape[1], image.shape[-1]))
                for i in range(ascent_image.shape[-1]):
                    ascent_image[:, :, i] = ascent.generate_ascent2D(image[:, :, i])
            elif image.ndim == 2:
                ascent_image = ascent.generate_ascent2D(image)
            images.append(ascent_image)
        except np.linalg.LinAlgError:
            print("Bad Image!")
            continue

    images = np.array(images)
    return images
