from imageio import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


def read_image(filename, representation):
    """
    open the given image with the given format
    Arguments:
        filename: image path
        representation: an integer for rgb or grey
    Returns:
        The image
    """
    im_g = imread(filename) / 255
    if representation == 1:
        im_g = rgb2gray(im_g)
    return im_g

def find_filter(filter_size):
    """
    create the filter vector according to the given size
    :param filter_size: the filter size
    :return: row vector used for the gaussian pyramid construction
    """
    if filter_size == 1:
        return np.array([[1]])
    basic_conv = np.array([1, 1]).reshape((1, 2))
    filter_vec = np.array([1, 1]).reshape((1, 2))
    for i in range(0, filter_size - 2):
        filter_vec = convolve2d(filter_vec, basic_conv, mode='full')
    return filter_vec


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function construct a gaussian pyramid
    :param im: a grayscale image with double values in [0; 1]
    :param max_levels:the maximal number of levels in resulting pyramid
    :param filter_size: the size of the Gaussian filter
    :return:the resulting pyramid and row vector of shape (1, filter_size)
            used for the gaussian pyramid construction
    """
    pyr = [im]
    max_levels -= 1
    img = im
    filter_vec = find_filter(filter_size)
    filter_vec = (1 / np.sum(filter_vec)) * filter_vec
    while max_levels > 0 and img.shape[0] >= 32 and img.shape[1] >= 32:
        conv_x = filter_vec
        conv_y = filter_vec.transpose()
        img = convolve2d(img, conv_x, mode="same")
        img = convolve2d(img, conv_y, mode="same")
        img = img[::2, ::2]
        pyr.append(img)
        max_levels -= 1
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function construct a laplacian pyramid
    :param im: a grayscale image with double values in [0; 1]
    :param max_levels:the maximal number of levels in resulting pyramid
    :param filter_size: the size of the laplacian filter
    :return:the resulting pyramid and row vector of shape (1, filter_size)
            used for the pyramid construction
    """
    gaussian_pyr, gaussian_filter_vec = build_gaussian_pyramid(im, max_levels,
                                                               filter_size)
    pyr = []
    for i in range(0, len(gaussian_pyr) - 1):
        img = expend(gaussian_pyr[i + 1], 2 * gaussian_filter_vec)
        img = gaussian_pyr[i] - img
        pyr.append(img)
    pyr.append(gaussian_pyr[-1])
    return pyr, gaussian_filter_vec


def expend(img, filter_vec):
    """
    This function expand the given image
    :param img: The image
    :param filter_vec: the filter of the expend
    :return: the expended image
    """
    zeros = np.zeros(2 * np.array(img.shape), dtype=img.dtype)
    zeros[::2, ::2] = img
    img = zeros
    conv_x = filter_vec
    conv_y = filter_vec.transpose()
    img = convolve2d(img, conv_x, mode="same")
    img = convolve2d(img, conv_y, mode="same")
    return img


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr:The Laplacian pyramid
    :param filter_vec:The filter of the laplacian
    :param coeff:list of scalars
    :return:the img reconstructed from the its Laplacian Pyramid.
    """
    img = lpyr[-1] * coeff[-1]
    for i in range(len(lpyr) - 1, 0, -1):
        new_img = expend(img, 2 * filter_vec)
        img = lpyr[i - 1] * coeff[i - 1] + new_img
    return img


def render_pyramid(pyr, levels):
    """
    This function create a big image with all the pyramid levels
    :param pyr: A pyramid (gaussian or laplacian)
    :param levels: number of levels to present
    :return: a big image with all the images in the pyramid
    """
    size_of_y = pyr[0].shape[0]
    size_of_x = 0
    for i in range(levels):
        size_of_x += pyr[i].shape[1]
    res = np.zeros((size_of_y, size_of_x), dtype=np.float64)
    cur_x = 0
    size_of_y *= 2
    for i in range(levels):
        x_size = cur_x + pyr[i].shape[1]
        max_v = np.max(pyr[i])
        min_v = np.min(pyr[i])
        pyr[i] = (pyr[i] - min_v) / (max_v - min_v)
        res[0:int(size_of_y / 2), cur_x:x_size] = pyr[i]
        size_of_y = size_of_y / 2
        cur_x = x_size
    return res


def display_pyramid(pyr, levels):
    """
    This function display the stacked pyramid image
    :param pyr: A pyramid (gaussian or laplacian)
    :param levels:number of levels to present
    """
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, cmap='gray')
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
    This function does pyramid blending
    :param im1: First grayscale image to be blended
    :param im2: Second grayscale image to be blended
    :param mask: A boolean mask containing which parts of im1 and im2 should
                appear in the resulting
    :param max_levels: The max level of the pyramid
    :param filter_size_im: The size of the Gaussian Filter which
            define the filter used in the construction of the Laplacian
            pyramid of im1 and im2.
    :param filter_size_mask: The size of the Gaussian filter which
            define the filter used in the construction of the Gaussian pyramid of
            mask.
    :return: The blended img
    """
    im1_lap, im1_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    im2_lap, im2_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    mask_gaus, mask_vec = build_gaussian_pyramid(mask.astype(np.float64),
                                                 max_levels, filter_size_mask)
    blend = []
    for i in range(len(mask_gaus)):
        new = mask_gaus[i] * im1_lap[i] + (1 - mask_gaus[i]) * im2_lap[i]
        blend.append(new)
    ones = np.ones(len(mask_gaus))
    new_im = laplacian_to_image(blend, im1_vec, ones)
    new_im = np.clip(new_im, 0, 1)
    return new_im


def blending_example(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
    Create the blending example
    :param im1: First grayscale image to be blended
    :param im2: Second grayscale image to be blended
    :param mask: A boolean mask containing which parts of im1 and im2 should
                appear in the resulting
    :param max_levels: The max level of the pyramid
    :param filter_size_im: The size of the Gaussian Filter which
            define the filter used in the construction of the Laplacian
            pyramid of im1 and im2.
    :param filter_size_mask: The size of the Gaussian filter which
            define the filter used in the construction of the Gaussian pyramid of
            mask.
    """
    new = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels,
                           filter_size_im, filter_size_mask)
    new1 = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, max_levels,
                            filter_size_im, filter_size_mask)
    new2 = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, max_levels,
                            filter_size_im, filter_size_mask)
    im_blend = np.dstack((new, new1, new2))
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)
    plt.show()


def blending_example1():
    im1 = read_image('externals/blackJoker.jpg', 2)

    im2 = read_image('externals/new3red.jpg', 2)
    mask = read_image('externals/jokerTestMask.jpg', 1)
    blending_example(im1, im2, mask.astype(np.bool_), 5, 5, 5)


def blending_example2():
    im1 = read_image('externals/blackEarth.jpg', 2)
    im2 = read_image('externals/newFed.jpg', 2)
    mask = read_image('externals/newtestBall.jpg', 1)
    blending_example(im1, im2, mask.astype(np.bool_), 5, 5, 5)


blending_example2()
blending_example1()
