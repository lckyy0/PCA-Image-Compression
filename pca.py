import sys
import lzma
import numpy as np
from PIL import Image

P = 8
VARIANCE = 0.99
LEVELS = 20


def to_patches(plane):
    h, w = plane.shape
    return plane.reshape(h // P, P, w // P, P).swapaxes(1, 2).reshape(-1, P * P)


def from_patches(patches, h, w):
    return patches.reshape(h // P, w // P, P, P).swapaxes(1, 2).reshape(h, w)


def compress_plane(plane):
    patches = to_patches(plane)

    mean = np.mean(patches, axis=0)
    centered_patches = patches - mean

    cov_matrix = np.cov(centered_patches, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    eigenvalues = np.flip(eigenvalues)
    eigenvectors = np.flip(eigenvectors, 1)

    k = 1
    if np.sum(eigenvalues) > 1e-9:
        evr = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(evr, VARIANCE) + 1
    eigenvectors = eigenvectors[:, :k]

    compress = centered_patches @ eigenvectors

    scale = np.abs(compress).max() / LEVELS or 1.0
    compress = np.round(compress / scale).astype(np.int8)

    return compress, eigenvectors.astype(np.float16), mean.astype(np.float16), np.float32(scale)


def decompress_plane(data, name, h, w):
    compress = data[name + "_c"].astype(np.float64) * data[name + "_s"]
    eigenvectors = data[name + "_b"].astype(np.float64)
    mean = data[name + "_m"].astype(np.float64)
    return from_patches(compress @ eigenvectors.T + mean, h, w)


def compress_image(src, dst):
    img_array = np.asarray(Image.open(src).convert("RGB"), dtype=np.float64)
    height, width = img_array.shape[:2]

    h = height + (-height) % (2 * P)
    w = width + (-width) % (2 * P)
    img_array = np.pad(img_array, ((0, h - height), (0, w - width), (0, 0)), mode="edge")

    red_plane = img_array[:, :, 0]
    green_plane = img_array[:, :, 1]
    blue_plane = img_array[:, :, 2]

    Y = 0.299 * red_plane + 0.587 * green_plane + 0.114 * blue_plane
    Cb = -0.168736 * red_plane - 0.331264 * green_plane + 0.5 * blue_plane + 128
    Cr = 0.5 * red_plane - 0.418688 * green_plane - 0.081312 * blue_plane + 128

    Cb = Cb.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    Cr = Cr.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    data = {"shape": np.array([height, width, h, w])}
    for name, plane in (("y", Y), ("cb", Cb), ("cr", Cr)):
        data[name + "_c"], data[name + "_b"], data[name + "_m"], data[name + "_s"] = compress_plane(plane)

    with lzma.open(dst, "wb") as f:
        np.savez(f, **data)


def decompress_image(src, dst):
    with lzma.open(src, "rb") as f:
        data = np.load(f)
        height, width, h, w = data["shape"]
        Y = decompress_plane(data, "y", h, w)
        Cb = decompress_plane(data, "cb", h // 2, w // 2)
        Cr = decompress_plane(data, "cr", h // 2, w // 2)

    Cb = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
    Cr = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)

    red_plane = Y + 1.402 * (Cr - 128)
    green_plane = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    blue_plane = Y + 1.772 * (Cb - 128)

    rgb = np.stack([red_plane, green_plane, blue_plane], axis=2)
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    Image.fromarray(rgb[:height, :width]).save(dst)


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("compress", "decompress"):
        sys.exit("usage: python pca.py compress|decompress input output")
    if sys.argv[1] == "compress":
        compress_image(sys.argv[2], sys.argv[3])
    else:
        decompress_image(sys.argv[2], sys.argv[3])
