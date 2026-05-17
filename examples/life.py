"""Conway's Game of Life — generator-based, runs a few steps."""

from itertools import product


def neighbors(cells):
    counts = {}
    for x, y in cells:
        for dx, dy in product((-1, 0, 1), repeat=2):
            if dx == 0 and dy == 0:
                continue
            counts[(x + dx, y + dy)] = counts.get((x + dx, y + dy), 0) + 1
    return counts


def step(cells):
    counts = neighbors(cells)
    out = set()
    for cell, n in counts.items():
        if n == 3 or (n == 2 and cell in cells):
            out.add(cell)
    for cell in cells:
        if cell not in counts:
            # alive but no neighbors → dies (counts.get returns 0)
            pass
    return out


def evolve(cells, steps):
    yield cells
    for _ in range(steps):
        cells = step(cells)
        yield cells


def render(cells, w, h):
    rows = []
    for y in range(h):
        row = "".join("#" if (x, y) in cells else "." for x in range(w))
        rows.append(row)
    return "\n".join(rows)


# Glider pattern
glider = {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)}

for i, gen in enumerate(evolve(glider, 4)):
    print(f"=== gen {i} ===")
    print(render(gen, 6, 6))
