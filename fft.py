import math
from typing import List
from collections.abc import Iterable
import numpy as np


def power_of_two_ceiling(n: int):
    """Returns the smallest power of two that 
    is not smaller than n."""

    i = 0
    while (1 << i) < n:
        i += 1
    return 1 << i

def fft(a: np.ndarray[complex], inv: bool=False) -> np.ndarray[complex]:
    """Computes the DFT of the given array."""

    n = len(a)

    assert power_of_two_ceiling(n) == n, "Input array size needs to be a power of two."
    
    if n == 1:
        return np.array(a, dtype=complex)
    
    evens_fft = fft(a[::2], inv)
    odds_fft = fft(a[1::2], inv)

    # The n-th root of unity
    if not inv:
        omega_n = math.e ** (2 * math.pi * complex(0, 1) * (1.0 / n))
    else:
        omega_n = math.e ** (2 * math.pi * complex(0, 1) * (-1.0 / n))
   
    # The wave function
    omega = np.array([omega_n ** i for i in range(n)], dtype=complex)

    mult = 1 if not inv else 0.5
    #mult = ((2 ** (0.5)) / 2.0)
    return mult * (
        np.concatenate((evens_fft, evens_fft)) + 
        (omega * np.concatenate((odds_fft, odds_fft)))
    )


def polynomial_multiply(a: np.ndarray[float], b: np.ndarray[float]) -> np.ndarray[float]:
    """Multiplies two polynomials using fft. The input lists represent 
    the coefficients of the polynomials starting from order 0.
    
    The coefficients of the multiplication of two polynomials is the convolution 
    of the coefficients of the input polynomials
    
    a * b = FFT^{-1}(FFT(a) _. FFT(b))
    where * is the convolution operator and _. is the element-wise multiplication 
    operator
    """

    # Bump the polynomial lengths to be equal to each another and be a power of two
    m = power_of_two_ceiling(2 * max(len(a), len(b)))
    a_complete = np.concatenate((a, [0] * (m - len(a))))
    b_complete = np.concatenate((b, [0] * (m - len(b))))

    # Step 1: Go to the ft space
    ft_a = fft(a_complete, inv=False)
    ft_b = fft(b_complete, inv=False)

    # Step 2: Elementwise multiply
    ft = ft_a * ft_b

    # Step 3: Go back to the original space
    return fft(ft, inv=True).real


def brute_polynomial_multiply(a: np.ndarray[float], b: np.ndarray[float]) -> np.ndarray[float]:
    """Multiplies two polynomials in brute force manner. 
    The input lists represent the coefficients of the
    polynomials starting from order 0."""

    m = power_of_two_ceiling(2 * max(len(a), len(b)))
    a_complete = np.concatenate((a, [0] * (m - len(a))))
    b_complete = np.concatenate((b, [0] * (m - len(b))))

    return np.convolve(a_complete, b_complete)

def polynomial_to_str(a: Iterable[float]) -> str:
    """Returns a string representation of the given 
    polynomial (list of coefficients starting from order 0)."""

    coefficients = [
        str(x) if i == 0 else f"{str(x)}X^{i}"
        for (i, x) in enumerate(a) if x != 0
    ]
    return " + ".join(coefficients)