# Data

The complete TransPolyE dataset is intentionally stored outside the Git repository.

## Expected layout

```text
data/
├── train/
│   ├── matfile/
│   │   ├── sample_000001.mat
│   │   └── ...
│   └── poly_data.txt
├── val/
│   ├── matfile/
│   └── poly_data.txt
├── test/
│   ├── matfile/
│   └── poly_data.txt
└── sample/
    └── example.mat
```

Each `.mat` file must contain `u_data.u0`, `u_data.u1`, and `u_data.u2`.

Each metadata sample line has the form

```text
filename class_id gamma1 gamma2 domain_id t_min t_max x_min x_max
```

The training metadata file contains an additional final line with the 12 normalization statistics used by the original experiment.

## Full dataset

For details on how the data was generated, please refer to this paper and our previous work：J. Comput. Phys. (2025) 539: 114243.