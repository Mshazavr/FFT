import random
import numpy as np
from utils import log_elapsed_time
from fft import polynomial_multiply, brute_polynomial_multiply, polynomial_to_str, fft

@log_elapsed_time
def test_trivial():
    """Simple test for fft polynomial multiplication."""

    pol_a = np.array([1,1], dtype=float)
    pol_b = np.array([1,1], dtype=float)

    print(f"\t({polynomial_to_str(pol_a)}) * ({polynomial_to_str(pol_b)}) = ")
    print(f"\t{polynomial_to_str(map(lambda x: round(x), polynomial_multiply(pol_a, pol_b)))}")


@log_elapsed_time
def test_basic():
    """Simple test for fft polynomial multiplication."""

    pol_a = np.array([4,1, -4, 1, 1], dtype=float)
    pol_b = np.array([4, 5, 0, -2], dtype=float)

    print(f"\t({polynomial_to_str(pol_a)}) * ({polynomial_to_str(pol_b)}) = ")
    print(f"\t{polynomial_to_str(map(lambda x: round(x), polynomial_multiply(pol_a, pol_b)))} VS")
    print(f"\t{polynomial_to_str(map(lambda x: round(x), brute_polynomial_multiply(pol_a, pol_b)))}")


@log_elapsed_time
def test_large(count: int = 50000, method: str = "fft"):
    """Test for fft polynomial multiplication with given input size."""

    assert method in {"fft", "brute_force"}
    random.seed(42)
    pol_a = np.array([random.random() for i in range(count)])
    pol_b = np.array([random.random() for i in range(count)])

    res = (
        polynomial_multiply(pol_a, pol_b) if method == "fft" 
        else brute_polynomial_multiply(pol_a, pol_b)
    )

    print(f"Error in constant term: {res[0] - pol_a[0] * pol_b[0]}")


@log_elapsed_time
def test_fft(count: int = 50000, method: str = "fft"):
    """Test for fft polynomial multiplication with given input size."""

    assert method in {"fft", "numpy_fft"}
    random.seed(42)
    a = np.array([random.random() for i in range(count)])

    if method == "fft":
        fft(a)
    else:
        np.fft.fft(a)

def main():
    test_trivial()
    test_basic()
    
    test_large(count=10000, method="fft")
    test_large(count=10000, method="brute_force")

    test_fft(count=(1 << 20), method="fft")
    test_fft(count=(1 << 20), method="numpy_fft")

if __name__ == "__main__":
    main()