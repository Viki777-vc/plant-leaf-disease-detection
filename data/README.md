# Dataset

Keep datasets out of Git history. Download the PlantVillage dataset from its original source and extract it locally. The training script expects one subfolder per class, for example:

```text
color/
├── Apple___Apple_scab/
├── Apple___Black_rot/
└── ...
```

Run training with:

```bash
python src/train.py --data_dir "path/to/color"
```

Do not upload the complete dataset archive or private test images to this repository. Add attribution and dataset terms if you redistribute any sample data.
