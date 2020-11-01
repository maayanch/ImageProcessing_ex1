import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

BINS = 256
MAX_COLOR_VAL = 255
GREY_SCALE_IM = 1
RGB_TO_YIQ = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]).T
RGB_IMAGE = 2


def read_image(filename, representation):
    """
    Function that reads an image file and converts it into a given representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be
    a grayscale image (1) or an RGB image (2). If the input image is grayscale, we won’t call it
    with representation = 2.
    :return:This function returns an image
    """
    check_representation(representation)
    im = imread(filename).astype(np.float64) / MAX_COLOR_VAL
    if representation == GREY_SCALE_IM:
        im = rgb2gray(im)
    return im


def check_representation(representation):
    if (representation != RGB_IMAGE and representation != GREY_SCALE_IM):
        print("for grayscle representation use 1, for RGB use 2")
        exit(1)


def imdisplay(filename, representation):
    """
    function that display an image in a given representation
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be
    a grayscale image (1) or an RGB image (2). If the input image is grayscale, we won’t call it
    with representation = 2.
    """
    check_representation(representation)
    im = read_image(filename, representation)
    cmap = 'gray' if representation == GREY_SCALE_IM else None
    plt.imshow(im, cmap=cmap)
    plt.show()


def rgb2yiq(imRGB):
    """
    Transforms the image from RGB space to YIQ space
    :param imRGB: Image to transform
    :return: Transformed image
    """
    return np.dot(imRGB, RGB_TO_YIQ)


def yiq2rgb(imYIQ):
    """
    Transforms the image from YIQ space to RGB space
    :param imRGB: Image to transform
    :return: Transformed image
    """
    return np.dot(imYIQ, np.linalg.inv(RGB_TO_YIQ))


def histogram_equalize(im_orig):
    """
    Performs histogram equalization of a given grayscale or RGB image.
    :param im_orig:The input grayscale or RGB float64 image with values in [0, 1].
    :return:a list [im_eq, hist_orig, hist_eq] where
        im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
        hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
        hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    img_greyscale = np.copy(im_orig)
    # check if the image in 3 or 1 dim. 3 means RGB image, else means greyscale image
    rgb_image = (len(im_orig.shape) == 3)

    img_greyscale = get_greyscale_img(img_greyscale, rgb_image)
    cumsum_hist, hist = get_cumsum_hist(img_greyscale)
    cumsum_hist_normalized = cumsum_hist / np.max(cumsum_hist)
    min_non_zero_val = get_first_non_zero_bin(cumsum_hist_normalized)
    equalized_hist = calc_hist_formula(cumsum_hist_normalized, min_non_zero_val)
    eq_im = get_final_img(equalized_hist, im_orig, img_greyscale, rgb_image)

    eq_im = np.clip(eq_im, 0, 1)
    return eq_im, hist, equalized_hist


def get_cumsum_hist(img_greyscale):
    """
    caculate histogram and cumsum histogram
    :param img_greyscale:
    :return:
    hist: is a 256 bin histogram of the original image
    cumsum hist
    """
    hist, bin_edges = np.histogram(img_greyscale, BINS, [0, BINS])
    cumsum_hist = np.cumsum(hist)
    return cumsum_hist, hist


def get_final_img(hist, im_orig, img_greyscale, rgb_image):
    """
    Map the intensity values of the image using the equlized hist
    :return: final image
    """
    # move each pixel intensity in the old image with the new intensity
    new_image = hist[img_greyscale.astype(np.int8)]
    new_image = new_image.astype(np.float64) / MAX_COLOR_VAL
    if rgb_image:
        yiq_image = rgb2yiq(im_orig)
        yiq_image[:, :, 0] = new_image[:, :]
        result_image = yiq2rgb(yiq_image)
    else:
        result_image = new_image
    return result_image


def get_first_non_zero_bin(cumsum_hist_normalized):
    """
    get the first value (bin) that not equal 0. The minimum val in the histogram
    :param cumsum_hist_normalized:
    :return: minimum value index
    """
    hist_ignored_zeros = np.ma.masked_equal(cumsum_hist_normalized, 0)
    min_index_not_zero = np.argmin(hist_ignored_zeros)
    min_not_zero_val = cumsum_hist_normalized[min_index_not_zero]
    return min_not_zero_val


def get_greyscale_img(img_greyscale, rgb_image):
    """
   transform only RGB image to grayscale image
    :param img_greyscale:
    :param rgb_image: boolean val, if RGB image equal True, else False
    :return: greyscale image
    """
    if rgb_image:
        img_greyscale = rgb2yiq(img_greyscale)
        img_greyscale = img_greyscale[:, :, 0]
    img_greyscale *= MAX_COLOR_VAL
    return img_greyscale


def calc_hist_formula(cumsum_hist_normalized, min_not_zero_val):
    """
    Calculate the formula of histogram
    :param cumsum_hist_normalized:
    :param min_not_zero_val:
    :return: equalized histogram by formula
    """
    non_zero_cumsum_hist = np.ma.masked_equal(cumsum_hist_normalized, 0)
    equalized_hist = np.round(((non_zero_cumsum_hist - min_not_zero_val) / (
            non_zero_cumsum_hist[MAX_COLOR_VAL] - min_not_zero_val)) * MAX_COLOR_VAL)

    equalized_hist = np.ma.filled(equalized_hist, 0)
    return equalized_hist


def get_init_z(img_greyscale, n_quant):
    cumsum_hist, hist = get_cumsum_hist(img_greyscale)
    segment_size = cumsum_hist[-1] / n_quant
    z = [0]
    z += [np.argmax(cumsum_hist >= segment_size * i) for i in range(1, n_quant)]
    z.append(MAX_COLOR_VAL)
    return z


def get_new_q(hist, z, q):
    for i in range(len(q)):
        z_i, z_i_next = get_zi_zinext(i, z)
        g, h_g = get_g_hg(hist, z_i, z_i_next)
        q[i] = (np.sum(h_g * g)) / (np.sum(h_g))
    return q


def get_new_z(q, z_list):
    for i in range(1, len(q)):
        if np.isnan(q[i]) or np.isnan(q[i - 1]):
            z_list[i] = 0
            continue
        z_list[i] = np.ceil((q[i - 1] + q[i]) / 2)
    return z_list


def compute_error(hist, z_list, q):
    err = 0
    for i in range(len(q)):
        z_i, z_i_next = get_zi_zinext(i, z_list)
        g, h_g = get_g_hg(hist, z_i, z_i_next)
        err += np.sum((np.square(q[i] - g)) * h_g)
    return err


def get_g_hg(hist, z_i, z_i_next):
    g = np.arange(z_i, z_i_next)
    h_g = hist[z_i:z_i_next]
    return g, h_g


def get_zi_zinext(i, z_list):
    z_i = int(z_list[i])
    z_i_next = int(z_list[i + 1])
    return z_i, z_i_next


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given grayscale or RGB image
    :param im_orig:the input grayscale or RGB image to be quantized (float64 image with
                   in [0, 1]).
    :param n_quant: the number of intensities your output im_quant image should have
    :param n_iter:s the maximum number of iterations of the optimization procedure.
    :return: a list [im_quant, error] where
            im_quant - is the quantized output image.
            error - is an array with shape (n_iter,) (or less) of the total intensities error
            for each iteration of the quantization procedure.
    """

    rgb_image = (len(im_orig.shape) == 3)
    img_greyscale = np.copy(im_orig)
    img_greyscale = get_greyscale_img(img_greyscale, rgb_image)
    hist, bin_edges = np.histogram(img_greyscale, BINS, [0, BINS])

    z_list = get_init_z(img_greyscale, n_quant)
    err = []
    q = np.zeros(n_quant)

    q, z_list, err = improve_resualts(err, hist, n_iter, q, z_list)

    for i in range(n_quant):
        z_i, z_i_next = get_zi_zinext(i, z_list)
        hist[z_i:z_i_next] = np.ceil(q[i])

    result_image = get_final_img(hist, im_orig, img_greyscale, rgb_image)
    return result_image, err


def improve_resualts(err, hist, n_iter, q, z_list):
    i = 0
    while i < n_iter:
        prev_z = np.copy(z_list)
        q = get_new_q(hist, z_list, q)
        z_list = get_new_z(q, z_list)
        if np.array_equal(z_list, prev_z):
            break
        error = compute_error(hist, z_list, q)
        err.append(error)
        i += 1
    return q, z_list, err


def quantize_rgb(im_orig, n_quant):
    """
    reduce color in rgb image
    :param im_orig:
    :param n_quant: num of colors
    :return: new image
    """
    w, h, d = tuple(im_orig.shape)
    im_arr = np.reshape(im_orig, (w * h, d))
    im_arr_sample = shuffle(im_arr, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_quant, random_state=0).fit(im_arr_sample)
    labels = kmeans.predict(im_arr)
    image = np.zeros((w, h, kmeans.cluster_centers_.shape[1]))
    lb = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = kmeans.cluster_centers_[labels[lb]]
            lb += 1
    return image


res = read_image("C:\\Users\\maayantz\\Desktop\\maytal\\yuv.jpeg", 1)
res, b,x = histogram_equalize(res)
res, err = quantize(res,2,10)
# res = quantize_rgb(im, 5)
plt.imshow(res,'gray')
plt.show()
