# A Phonotactic Atlas of the World's Languages

## Overview

This project uses pronunciation data from [WikiPron](https://github.com/CUNY-CL/wikipron) to test whether the words of a language share common sound patterns, and whether those patterns are shared across languages. It applies several techniques and measures to the data and finds that languages that are closer in the language family tend to sound more similar.

To go beyond family relationships, the project also uses [Glottolog](https://glottolog.org/) to measure each language's distance from its origin. The results suggest that geographic proximity, in addition to family membership, helps explain why some languages sound alike.

## Repository Structure

```
.
├── data/                  # Raw and processed data (WikiPron, Glottolog)
├── src/                   # Reusable modules imported by the scripts
│   ├── .py        # <what it does, e.g. loads and cleans WikiPron data>
│   └── <module>.py        # <e.g. computes pairwise phonotactic distances>
├── scripts/               # Entry-point scripts, run in order
│   ├── 01_<name>.py       # <e.g. build phonotactic profiles per language>
│   ├── 02_<name>.py       # <e.g. compute distances between languages>
│   ├── 03_<name>.py       # <e.g. generate maps>
│   └── 04_<name>.py       # <e.g. test family and geography effects>
├── results/               # Output tables and figures
└── requirements.txt
```

### Source Code (`src/`)

| File | Purpose |
|------|---------|
| `<module>.py` | <One sentence on what it does and its main functions> |
| `<module>.py` | <...> |

### Scripts (`scripts/`)

Run the scripts in numerical order. Each reads the output of the previous one.

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `01_<name>.py` | WikiPron TSVs | `data/profiles.csv` | <...> |
| `02_<name>.py` | `profiles.csv` | `distances.csv` | <...> |
| `03_<name>.py` | `distances.csv`, Glottolog | `results/maps/` | <...> |
| `04_<name>.py` | `distances.csv`, Glottolog | `results/tests.csv` | <...> |

## Usage

```bash
pip install -r requirements.txt
python scripts/01_<name>.py
python scripts/02_<name>.py
# ...
```
