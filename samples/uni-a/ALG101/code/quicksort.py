import random


def quicksort(xs: list[int]) -> list[int]:
    """Randomised quicksort: expected O(n log n)."""
    if len(xs) <= 1:
        return xs
    pivot = random.choice(xs)
    smaller = [x for x in xs if x < pivot]
    equal = [x for x in xs if x == pivot]
    larger = [x for x in xs if x > pivot]
    return quicksort(smaller) + equal + quicksort(larger)
