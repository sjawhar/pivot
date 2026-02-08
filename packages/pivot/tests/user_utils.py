def leaf_func(x: int) -> int:
    return x * 2


def helper_a(x: int) -> int:
    return leaf_func(x) + 1


def helper_b(x: int) -> int:
    return helper_a(x) + 10


CONSTANT_A = 100


def unused_func(x: int) -> int:
    return x * 999
