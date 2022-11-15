import matplotlib.pyplot as plt
from fft import fft, power_of_two_ceiling
import copy
import numpy as np

from im_compressor import encode_image, decode_image

def main():
    img = plt.imread("data/dog1.jpeg", format="jpeg")

    for i in range(10):
        compression_rate = 10 * (i + 1)
        encoded_object = encode_image(img, compression_rate=compression_rate)
        decoded_image = decode_image(encoded_object)
        plt.imsave(f"data/dog1_q{compression_rate}.jpeg", decoded_image, format="jpeg")


if __name__ == "__main__":
    main()