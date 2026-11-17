# Sample data

This directory is intended for a small representative subset copied from the original **test** set.

Recommended public sample: one sample per class per observation domain. With six classes and three domains, this gives 18 MATLAB files. If the resulting folder is too large for GitHub, use one observation domain only (six files).

Create the subset from the full dataset with:

```bash
python tools/make_sample_dataset.py \
  --source-data /path/to/full/test/matfile \
  --source-list /path/to/full/test/poly_data.txt \
  --output data/sample \
  --samples-per-class 1
```

The script copies existing `.mat` files and writes a matching `poly_data.txt`; it does not regenerate data or retrain the network.
