def fibonacci(n_terms):
    """
    Generate Fibonacci sequence up to n terms
    
    Args:
        n_terms (int): Number of terms to generate
        
    Returns:
        list: Fibonacci sequence as a list
    """
    if n_terms <= 0:
        return []
    elif n_terms == 1:
        return [0]
    
    sequence = [0, 1]
    
    while len(sequence) < n_terms:
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)
    
    return sequence

# Example usage
if __name__ == "__main__":
    num_terms = 10
    fib_sequence = fibonacci(num_terms)
    print(f"First {num_terms} Fibonacci numbers:")
    print(fib_sequence)