"""Best-fit allocation for a static DAG, with birth-before-last-use semantics."""
from math import prod


def layout(nodes):
    last = list(range(len(nodes)))
    for i, n in enumerate(nodes):
        for parent in n.parents:
            last[parent] = i
    offsets, active = [], {}
    peak = lower = 0
    for i, n in enumerate(nodes):
        active = {j:v for j,v in active.items() if last[j] >= i}
        size = prod(n.shape)
        occupied = sorted(active.values())
        cursor = 0
        gaps = []
        for start, count in occupied:
            if start - cursor >= size:
                gaps.append((start-cursor, cursor))
            cursor = start+count
        if peak-cursor >= size:
            gaps.append((peak-cursor, cursor))
        offset = min(gaps)[1] if gaps else cursor
        offsets.append(offset)
        active[i] = (offset, size)
        peak = max(peak, offset+size)
        lower = max(lower, sum(v[1] for v in active.values()))
    return offsets, peak, lower
