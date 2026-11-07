# PCA Image Compression

A tiny image compressor made with just NumPy and Pillow, using patch-based PCA and YCbCr.

## Usage

```
pip install numpy
pip install pillow

python pca_codec.py compress lena.jpg lena.pca
python pca_codec.py decompress lena.pca restored.png
```

## How it works

```
RGB -> YCbCr -> halve Cb and Cr -> cut each channel into 8x8 patches
    -> PCA per channel -> keep top k components -> round to int8 -> lzma
```

- **YCbCr** puts brightness in Y and colour in Cb/Cr. Eyes care less about colour detail, so Cb and Cr are shrunk to half size.
- **Patches** turn each channel into rows of 64 numbers. Pixels in a small block are very similar, so a few components describe a block well.
- **PCA** per channel: subtract the mean patch, take the covariance matrix, eigendecompose it, and keep the smallest k that holds `VARIANCE` of the total variance.

$$Z = (P - \mu)\,Q_k \qquad\qquad \hat P = Z\,Q_k^T + \mu$$

- **Quantization** divides the coefficients by one step per channel (`max / LEVELS`) and rounds to int8.
- **lzma** compresses the whole saved file.

## Settings

Two constants at the top of `pca.py`:

| Constant | Default | Effect |
|---|---|---|
| `VARIANCE` | `0.99` | Higher keeps more components: bigger file, sharper image |
| `LEVELS` | `20` | Lower means a coarser step: smaller file, a bit more error |


## Notes

- Any image size works. Edges are padded, then cropped on decompress.
- The `.pca` file is an lzma-compressed NumPy `.npz`.
- Each patch is coded on its own, so blocky edges can show up at low settings.
