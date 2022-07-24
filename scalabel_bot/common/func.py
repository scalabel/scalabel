def cantor_pairing(x: int, y: int):
    """Cantor pairing function."""
    return (x + y) * (x + y + 1) // 2 + y
