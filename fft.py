import math
import random
import time
from typing import Callable, List

def log_elapsed_time(func: Callable):
    """Decorator for timing a function."""

    def wrapper_log_elapsed_time(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"ELAPSED TIME: {end - start}")
        return res
    return wrapper_log_elapsed_time

def power_of_two_ceiling(n: int):
    """Returns the smallest power of two that 
    is not smaller than n."""

    i = 0
    while (1 << i) < n:
        i += 1
    return 1 << i

def fft(a: List[complex], inv: bool=False) -> List[complex]:
    """Computes the DFT of the given array."""

    n = len(a)
    
    if n == 1:
        return a
    
    evens_fft = fft([x for (i,x) in enumerate(a) if i % 2 == 0], inv)
    odds_fft = fft([x for (i,x) in enumerate(a) if i % 2 == 1], inv)

    # The n-th root of unity
    if not inv:
        omega_n = math.e ** (2 * math.pi * complex(0, 1) * (1.0 / n))
    else:
        omega_n = math.e ** (2 * math.pi * complex(0, 1) * (-1.0 / n))
    
    mult = 1 if not inv else 0.5
    return [
        (evens_fft[i % int(n/2)] + (omega_n ** (i)) * odds_fft[i % int(n/2)]) * mult
        for i in range(n)
    ]

def polynomial_multiply(a: List[float], b: List[float]) -> List[float]:
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
    a_complete = a + [0] * (m - len(a))
    b_complete = b + [0] * (m - len(b))

    # Step 1: Go to the ft space
    ft_a = fft(a_complete, inv=False)
    ft_b = fft(b_complete, inv=False)

    # Step 2: Elementwise multiply
    ft = [x * y for (x, y) in zip(ft_a, ft_b)]

    # Step 3: Go back to the original space
    return map(lambda x : round(x.real), fft(ft, inv=True))


def brute_polynomial_multiply(a: List[float], b: List[float]) -> List[float]:
    """Multiplies two polynomials in brute force manner. 
    The input lists represent the coefficients of the
    polynomials starting from order 0."""

    m = power_of_two_ceiling(2 * max(len(a), len(b)))
    a_complete = a + [0] * (m - len(a))
    b_complete = b + [0] * (m - len(b))

    c = [0] * m
    for i in range(m):
        for j in range(i+1):
            c[i] += a_complete[j] * b_complete[i-j]

    return c



def polynomial_to_str(a: List[float]) -> str:
    """Returns a string representation of the given 
    polynomial (list of coefficients starting from order 0)."""

    coefficients = [
        str(x) if i == 0 else f"{str(x)}X^{i}"
        for (i, x) in enumerate(a) if x != 0
    ]
    return " + ".join(coefficients)

def test_1():
    pol_a = [1,1]
    pol_b = [1,1]

    print (f"({polynomial_to_str(pol_a)}) * ({polynomial_to_str(pol_b)}) = ")
    print(f"{polynomial_to_str(polynomial_multiply(pol_a, pol_b))}")


def test_2():
    pol_a = [4,1, -4, 1, 1]
    pol_b = [4, 5, 0, -2]

    print (f"({polynomial_to_str(pol_a)}) * ({polynomial_to_str(pol_b)}) = ")
    print(f"{polynomial_to_str(polynomial_multiply(pol_a, pol_b))}")
    print ("VS")
    print(f"{polynomial_to_str(brute_polynomial_multiply(pol_a, pol_b))}")


# Should run under 3 seconds
@log_elapsed_time
def test_3():
    random.seed(42)
    pol_a = [random.random() for i in range(50000)]
    pol_b = [random.random() for i in range(50000)]

    start = time.time()
    res = polynomial_multiply(pol_a, pol_b)
    #res = brute_polynomial_multiply(pol_a, pol_b)

def main():
    test_1()
    test_2()
    test_3()

if __name__ == "__main__":
    main()