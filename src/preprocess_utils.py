import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

# The path to the data relative to the src folder
images_path = '../data/images/'
masks_path = '../data/masks/'

# File type endings for images and masks, respectively
image_post = '.HW.K.P.rdc.fits.gz'
mask_post = '.HW.M.P.rdc.fits.gz'

# Preprocessing Functions

def tiler(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def fits_processor(image_name, mask_name, keepall=True, keepspace=False):
    """
    Takes an image FITS file and a mask FITS file, extracts their data to arrays,
    makes the mask array binary for if a pixel is a sunspot or not,
    breaks each 2048x2048 into 256 128x128 tiles, and then returns a list of tuple
    triplets of corresponding image and mask tiles and a filename stub.
    If keepall is False, only tiles where the sum of the mask tile is greater than
    zero are kept. If keepspace is False, only tiles that contain at least some
    of the disk of the sun are kept.
    """

    image_hdu = fits.open(image_name)
    mask_hdu = fits.open(mask_name)

    image = image_hdu[0].data.copy()
    mask = mask_hdu[0].data.copy()
    mask[0,0] = 0

    image_hdu.close()
    mask_hdu.close()

    image_tiles = tiler(image, 128, 128)
    mask_tiles = tiler(mask, 128, 128)

    tile_pairs = []

    for i in xrange(256):
        # Skip tiles that contain only space if keepspace = False
        if (not keepspace) and (mask_tiles[i].sum() == 0):
            continue

        current_mask = np.zeros((128,128))
        current_mask[mask_tiles[i] > 6] = 1

        if keepall:
            tile_pairs.append((image_tiles[i], current_mask, image_name[-32:-19]+'.'+str(i)))
            continue

        if current_mask.sum() > 1:
            tile_pairs.append((image_tiles[i], current_mask, image_name[-32:-19]+'.'+str(i)))

    return tile_pairs

def write_arrays(filepath, pairs):
    """
    WARNING: this function assumes 2 folders named image_arrays and mask_arrays
    in the filepath provided.

    Takes a list of tuple triplets of corresponding image and mask tiles and a
    filename stub and saves image tiles to filepath/image_arrays, and mask tiles to
    filepath/mask_arrays.
    """

    image = filepath + '/image_arrays/'
    mask  = filepath + '/mask_arrays/'

    for i, pair in enumerate(pairs):
        np.save(image+pair[2], pair[0])
        np.save(mask+pair[2], pair[1])

def write_binary_png(filepath, pairs):
    """
    WARNING: this function assumes 2 folders named 0 and 1 are in the filepath provided.

    Takes a list of tuple triplets of corresponding image and mask tiles and a
    filename stub and saves image tiles that contain sunspots to filepath/1,
    and images that don't contain sunspots to filepath/0 as a png.
    """

    yes = filepath + '/1/'
    no  = filepath + '/0/'

    for i, pair in enumerate(pairs):
        if pair[1].sum():
            plt.imsave(yes+pair[2]+'.png', pair[0])
            continue
        plt.imsave(no+pair[2]+'.png', pair[0])

def populate_binary(filepath, inter_lst, keepall, start, stop):
    """
    Takes a filepath, a sorted list FITS filename stubs, the keepall flag, a start index,
    and a stop index. Populates filepath with image PNGs sorted into folders denoting
    if the image contains a sunspot or not.
    """
    for i in xrange(start, stop):
        if not i%50:
            print i

        write_binary_png(filepath, fits_processor(images_path+inter_lst[i]+image_post,
            masks_path+inter_lst[i]+mask_post, keepall))

def populate_arrays(filepath, inter_lst, start, stop):
    """
    Takes a filepath, a sorted list of FITS filename stubs, a start index, and a stop index.
    Populates the filepath/image_arrays and filepath/mask_arrays folders with image
    and mask tiles as .npy files, respectively.
    """

    for i in xrange(start, stop):
        if not i%50:
            print i

        write_arrays(fits_processor(images_path+inter_lst[i]+image_post,
            masks_path+inter_lst[i]+mask_post))
