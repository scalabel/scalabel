def cantor_pairing(x_1: int, x_2: int) -> int:
    """Cantor pairing function."""
    return (x_1 + x_2) * (x_1 + x_2 + 1) // 2 + x_2
