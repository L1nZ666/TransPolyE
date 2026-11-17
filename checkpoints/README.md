# Checkpoints

Place the trained model checkpoint here:

```text
checkpoints/best_model_multidomain.pth
```

The original checkpoint is expected to contain at least:

- `model_state_dict`
- `Nx`
- `Nt`
- `stats`

The cleaned training script additionally stores a `model_config` dictionary.

TODO: add the public checkpoint DOI/URL after release.
