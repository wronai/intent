def fibonacci(n):
    """
    Calculate the nth Fibonacci number recursively.

    Args:
        n (int): The position of the Fibonacci number to calculate.

    Returns:
        int: The nth Fibonacci number.

    Raises:
        ValueError: If n is a negative integer.
    """

    # Check if input is an integer
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")

    # Base cases for recursion
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    elif n == 1:
        return 0
    elif n == 2:
        return 1

    # Recursive case
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
