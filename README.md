# Assignment 2

## Minimal Folder Structure

```text
Assignment 2/
├─ README.md
├─ DL_project.pdf
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ src/
└─ reports/
   └─ figures/
```

## What Goes Where

- `data/raw/`: original datasets or untouched downloads.
- `data/processed/`: cleaned data that is ready for training or analysis.
- `notebooks/`: exploration, debugging, and quick experiments.
- `src/`: reusable code that we want to keep out of notebooks.
- `reports/figures/`: plots and figures for the report.

## Team Workflow

1. Save raw data separately from processed data so preprocessing stays reproducible.
2. Put final plots in `reports/figures/` and avoid scattering them across sources.
3. Avoid editing the same notebook or script at the same time without coordination.

## Before You Start Working

- Check `DL_project.pdf` for the assignment requirements.
