# %%
def largest_prime_factor(n: int) -> int:
    # Initialize the largest prime factor
    largest_prime = 0
    
    # While n is divisible by 2, divide it by 2 and update largest prime factor
    while n % 2 == 0:
        largest_prime = 2
        n //= 2

    # Check for odd prime factors starting from 3
    for i in range(3, int(n**0.5) + 1, 2):
        while n % i == 0:
            largest_prime = i
            n //= i
    
    # If n is still greater than 2, then it's a prime number itself
    if n > 2:
        largest_prime = n
    
    return largest_prime

def is_small_prime_factor(n: int) -> bool:
    """Checks if a number has only small prime factors (2, 3, or 5)."""
    if n < 2:
        raise ValueError("Input must be a positive integer greater than 1.")
    return n % 2 == 0 or n % 3 == 0 or n % 5 == 0

def nearest_number_with_small_prime_factors(n: int) -> int:
    """Returns the nearest number with only small prime factors (2, 3, or 5).
    This function is useful for choosing the real space grid size for exact scattering calculations.
    integers chosen is suitable for FFTs. 

    Args:
        n (int): an integer greater than 1

    Raises:
        ValueError: if n is less than 2

    Returns:
        int: the nearest number with only small prime factors
    """
    if n < 2:
        raise ValueError("Input must be a positive integer greater than 1.")
    max_factor = largest_prime_factor(n)
    if max_factor <= 5:
        return n
    else:
        return nearest_number_with_small_prime_factors(n + 1)
# %%
