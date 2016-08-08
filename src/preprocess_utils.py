import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os

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

def write_binary(pairs):
    """
    WARNING: this function assumes a file directory layout of being run from the src
    folder, and there being a data/binary folder with 2 folders names 0 and 1 in it.

    Takes a list of tuple triplets of corresponding image and mask tiles and a
    filename stub and saves image tiles that contain sunspots to data/binary/1,
    and images that don't contain sunspots to data/binary/0 as a npy.
    """

    yes = '../data/binary/1/'
    no  = '../data/binary/0/'

    for i, pair in enumerate(pairs):
        if pair[1].sum():
            np.save(yes+pair[2], pair[0])
            continue
        np.save(no+pair[2], pair[0])

def write_binary_png(filepath, pairs):
    """
    WARNING: this function creates 2 folders in the filepath provided.

    Takes a list of tuple triplets of corresponding image and mask tiles and a
    filename stub and saves image tiles that contain sunspots to filepath/1,
    and images that don't contain sunspots to filepath/0 as a png.
    """

    yes = filepath + '/1/'
    no  = filepath + '/0/'

    #os.mkdir(yes)
    #os.mkdir(no)

    for i, pair in enumerate(pairs):
        if pair[1].sum():
            plt.imsave(yes+pair[2]+'.png', pair[0])
            continue
        plt.imsave(no+pair[2]+'.png', pair[0])
