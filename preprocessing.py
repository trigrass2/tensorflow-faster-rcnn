import cv2


def process_image(im, proposal):
    max_size = proposal.max_size
    target_size = proposal.scales
    image_means = proposal.image_means

    im = im - image_means
    h, w, _ = im.shape
    im_scale = prep_im_scale([h, w], target_size, max_size)

    nrow, ncol = int(h * im_scale), int(w * im_scale)
    im = cv2.resize(im, (ncol, nrow))

    return im, im_scale


def prep_im_scale(im_size, target_size, max_size):
    im_min_size, im_max_size = min(im_size), max(im_size)
    im_scale = float(target_size) / im_min_size

    if round(im_scale * im_max_size) > max_size:
        im_scale = float(max_size) / im_max_size

    return im_scale
