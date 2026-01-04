def leaf_func(x):
    return x * 2


def helper_a(x):
    return leaf_func(x) + 1


def helper_b(x):
    return helper_a(x) + 10


CONSTANT_A = 100


def unused_func(x):
    return x * 999
