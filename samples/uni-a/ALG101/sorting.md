# Sorting algorithms

Merge sort splits the array in halves, sorts each half recursively and merges them.
It runs in O(n log n) time in every case and is stable, but needs O(n) extra memory.

Quicksort picks a pivot and partitions the array into elements smaller and larger than the pivot.
Average time is O(n log n); worst case O(n^2) happens when the pivot is always the smallest or largest element.
Randomised pivot choice makes the worst case unlikely.

Insertion sort is O(n^2) but fast on small or nearly sorted inputs.
