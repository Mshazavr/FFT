import matplotlib.pyplot as plt
from fft import fft, power_of_two_ceiling
import copy
import numpy as np

from im_compressor import encode_image, decode_image

def main():
    img = plt.imread("data/cat.jpg", format="jpg")

    for i in range(10):
        compression_rate = 10 * (i + 1)
        encoded_object = encode_image(img, compression_rate=compression_rate)
        decoded_image = decode_image(encoded_object)
        plt.imsave(f"data/compressed/cat1_q{compression_rate}.jpg", decoded_image, format="jpg")


if __name__ == "__main__":
    main()