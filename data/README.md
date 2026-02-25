# Datasets

This directory stores HDF5 dataset files downloaded from [ann-benchmarks.com](http://ann-benchmarks.com/).

## Supported Datasets

| Dataset | Dimensions | Base Vectors | Query Vectors | Metric |
|---------|-----------|-------------|---------------|--------|
| sift-128-euclidean | 128 | 1,000,000 | 10,000 | L2 |
| glove-100-angular | 100 | 1,183,514 | 10,000 | Angular/Cosine |

## Download

Run the download script:
```bash
python data/download_datasets.py
```

Or download manually from http://ann-benchmarks.com/ and place `.hdf5` files here.
