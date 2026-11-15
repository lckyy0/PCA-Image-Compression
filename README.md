# PCA Image Compression

A tiny image compressor made with just NumPy and Pillow, using patch-based PCA and YCbCr.

## Usage

```
pip install numpy pillow

python pca.py compress photo.jpg photo.pca
python pca.py decompress photo.pca restored.png
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

- **Quantization** divides the coefficients by one step per channel (`max / LEVELS`) and rounds to int8. Trailing components that round to zero are dropped.
- **lzma** compresses the whole saved file.

## Settings

Three constants at the top of `pca.py`:

| Constant | Default | Effect |
|---|---|---|
| `P` | `8` | Patch side length. Padding is a multiple of `2 * P` |
| `VARIANCE` | `0.99` | Higher keeps more components, but ones too small for the `LEVELS` step are dropped anyway |
| `LEVELS` | `20` | Main quality knob. Lower means a coarser step: smaller file, more error. Max 127 |


## Notes

- Any image size works. Edges are padded to a multiple of 16, then cropped on decompress.
- The `.pca` file is an lzma-compressed NumPy `.npz`. Basis and mean are stored as float16.
- Output is 8-bit RGB. Alpha and colour profiles are dropped, and EXIF rotation is applied.
- Only open `.pca` files you made yourself.
- Each patch is coded on its own, so blocky edges can show up at low settings.
