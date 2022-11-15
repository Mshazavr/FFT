from dataclasses import dataclass
import numpy as np
from fft import fft, power_of_two_ceiling

@dataclass
class ImageFFTCompressedData:
    """Class for storing information of fft-compressed image."""

    height: int
    width: int 
    fft_height: int 
    fft_width: int 
    fft_red_grayscales: np.ndarray[complex]
    fft_green_grayscales: np.ndarray[complex]
    fft_blue_grayscales: np.ndarray[complex]


def decode_image(fft_object: ImageFFTCompressedData) -> np.ndarray[complex]:
    """Given a compressed image, returns numpy array representing 
    a low-quality version of the image."""

    def decode_grayscale(grayscale: np.ndarray[complex], final_height: int, final_width: int) -> np.ndarray[int]:
        # Inverse transform columns and truncate
        result = np.array(
            [
                fft(col, inv=True) 
                for col in grayscale.transpose()
            ]
        ).transpose()[:final_height, :]

        # Inverse transform rows and truncate and convert to int
        result = np.array(
            [fft(row, inv=True) for row in result]
        )[:, :final_width].real.astype("uint8")

        return result

    return np.dstack(
        (
            decode_grayscale(fft_object.fft_red_grayscales, fft_object.height, fft_object.width),
            decode_grayscale(fft_object.fft_green_grayscales, fft_object.height, fft_object.width),
            decode_grayscale(fft_object.fft_blue_grayscales, fft_object.height, fft_object.width),
        )
    )


def encode_image(image: np.ndarray[int], compression_rate: int = 80) -> ImageFFTCompressedData:
    """Given a 3d numpy array representing an image (with dimensions
    for row, column and RGB axis), returns a compressed representation 
    of the image."""

    height = len(image)
    width = len(image[0])

    fft_height = power_of_two_ceiling(height)
    fft_width = power_of_two_ceiling(width)

    def filter_by_absolute_value(a: np.ndarray[complex]) -> np.ndarray[complex]:
        """Given a complex ndarray, turns the entries with small amplitute
        zero according to the compression rate. Namely, compression_rate% of the
        values turn to zero."""

        threshold = np.percentile(np.abs(a), compression_rate)
        a[np.abs(a) < threshold] = 0
        return a

    def encode_grayscale(grayscale: np.ndarray[int]) -> np.ndarray[int]:
        # Pad the input to have power of two dimensions
        result = np.pad(grayscale, [[0, fft_height - height], [0, fft_width - width]])
        
        # Transform rows
        result = np.array([fft(row) for row in result])
        
        # Transform Columns
        result = np.array(
            [fft(col) for col in result.transpose()]
        ).transpose()

        return filter_by_absolute_value(result)

    return ImageFFTCompressedData(
        height=height,
        width=width,
        fft_height=fft_height,
        fft_width=fft_width,
        fft_red_grayscales=encode_grayscale(image[:, :, 0]),
        fft_green_grayscales=encode_grayscale(image[:, :, 1]),
        fft_blue_grayscales=encode_grayscale(image[:, :, 2])
    )