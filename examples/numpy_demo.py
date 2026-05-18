"""numpy demo: linear regression, image-like blur, k-means clustering.

Run the original:
    python examples/numpy_demo.py
or onexpr-obfuscated:
    python onexpr.py --input examples/numpy_demo.py --output obf.py
    python obf.py
"""

import numpy as np


def linreg(x, y):
    """Closed-form ordinary least squares: returns (slope, intercept)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.stack([x, np.ones_like(x)], axis=1)
    # Manually compute (A^T A)^-1 A^T y instead of np.linalg.lstsq
    # so the obfuscated path exercises matrix algebra.
    AtA = A.T @ A
    Aty = A.T @ y
    coeffs = np.linalg.solve(AtA, Aty)
    return float(coeffs[0]), float(coeffs[1])


def gaussian_blur(arr, sigma=1.0, radius=2):
    """1D gaussian blur applied along axis 0 then axis 1."""
    arr = np.asarray(arr, dtype=float)
    k = np.arange(-radius, radius + 1)
    weights = np.exp(-(k ** 2) / (2.0 * sigma * sigma))
    weights /= weights.sum()

    def blur1d(a, axis):
        # Reflect-pad along the requested axis, then sum the shifted
        # contributions weighted by the gaussian kernel.
        pad = [(0, 0)] * a.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(a, pad, mode="reflect")
        out = np.zeros_like(a)
        for i, w in enumerate(weights):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += w * padded[tuple(sl)]
        return out

    return blur1d(blur1d(arr, axis=0), axis=1)


def kmeans(points, k, iters=20, seed=0):
    """Returns (labels, centers) using Lloyd's algorithm."""
    rng = np.random.default_rng(seed)
    pts = np.asarray(points, dtype=float)
    # Pick k initial centers without replacement.
    init = rng.choice(len(pts), size=k, replace=False)
    centers = pts[init].copy()
    labels = np.zeros(len(pts), dtype=int)

    for _ in range(iters):
        # Squared distance from each point to each center, vectorized.
        d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Recompute centers; if a cluster is empty, leave its center.
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = pts[mask].mean(axis=0)
    return labels, centers


# --- linreg demo ---
np.random.seed(42)
true_slope, true_intercept = 2.0, 5.0
xs = np.linspace(0, 10, 50)
noise = np.random.normal(scale=0.3, size=xs.shape)
ys = true_slope * xs + true_intercept + noise

slope, intercept = linreg(xs, ys)
print(f"linreg: slope={slope:.3f} (true 2.0), intercept={intercept:.3f} (true 5.0)")

# --- blur demo: deterministic checkerboard ---
img = np.zeros((8, 8))
img[::2, ::2] = 1.0
img[1::2, 1::2] = 1.0
blurred = gaussian_blur(img, sigma=1.5, radius=3)
print("img total:", img.sum(), "blurred total:", round(float(blurred.sum()), 4))
print("blurred[3,3]:", round(float(blurred[3, 3]), 4))

# --- kmeans demo: three clusters in 2D ---
np.random.seed(0)
n = 60
g1 = np.random.normal(loc=(0, 0), scale=0.3, size=(n, 2))
g2 = np.random.normal(loc=(3, 0), scale=0.3, size=(n, 2))
g3 = np.random.normal(loc=(1.5, 3), scale=0.3, size=(n, 2))
data = np.vstack([g1, g2, g3])

labels, centers = kmeans(data, k=3, iters=30, seed=1)
# Sort centers by x for stable output.
order = np.argsort(centers[:, 0])
sorted_centers = centers[order]
print("centers:")
for c in sorted_centers:
    print(f"  ({c[0]:.2f}, {c[1]:.2f})")
print("cluster sizes:", sorted(np.bincount(labels).tolist()))
