import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS
import umap

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from src import config
from src.io_utils import load_glottolog

ap = argparse.ArgumentParser()
ap.add_argument("--level", choices=["class", "segment"], required=True)
ap.add_argument("--top", type=int, default=12, help="families shown in color")
args = ap.parse_args()

D = np.load(config.PROCESSED / f"dist_{args.level}_combined.npy")
meta = pd.read_parquet(config.PROCESSED / f"metadata_aligned_{args.level}.parquet")

# readable family names: look up each Family_ID in the Glottolog table
glot = load_glottolog(config.GLOTTOLOG_CSV)
names = dict(zip(glot["ID"], glot["Name"]))
meta["family_name"] = meta["family"].map(names).fillna(meta["family"])

# color only the biggest families, the rest are "other"
top = meta["family_name"].value_counts().head(args.top).index
meta["family_plot"] = meta["family_name"].where(meta["family_name"].isin(top), "other")

# two embeddings of the same distance matrix
mds = MDS(n_components=2, dissimilarity="precomputed",
          random_state=config.SEED, n_init=4, normalized_stress="auto")
xy_mds = mds.fit_transform(D)
xy_umap = umap.UMAP(n_components=2, metric="precomputed",
                    n_neighbors=config.UMAP_NEIGHBORS,
                    random_state=config.SEED).fit_transform(D)
print("MDS stress:", mds.stress_)

for tag, xy in [("mds", xy_mds), ("umap", xy_umap)]:
    meta[f"{tag}_x"], meta[f"{tag}_y"] = xy[:, 0], xy[:, 1]
    fig = px.scatter(meta, x=f"{tag}_x", y=f"{tag}_y", color="family_plot",
                     hover_name="Name", hover_data=["iso", "family_name"],
                     title=f"Sound map ({tag.upper()}, {args.level} level)")
    fig.write_html(config.RESULTS / f"map_{tag}_{args.level}.html")
    cmap = dict(zip(top, px.colors.qualitative.Dark24))
    cmap["other"] = "lightgrey"
    fig = px.scatter(meta, x=f"{tag}_x", y=f"{tag}_y", color="family_plot",
                     color_discrete_map=cmap, hover_name="Name",
                     hover_data=["iso", "family_name"],
                     title=f"Sound map ({tag.upper()}, {args.level} level)")
    fig.write_html(config.RESULTS / f"map_{tag}_{args.level}.html")

# world map: color each language by its position in the sound embedding
c = xy_mds[:, 0]
meta["sound_axis1"] = (c - c.min()) / (c.max() - c.min())
fig = px.scatter_geo(meta, lat="Latitude", lon="Longitude",
                     color="sound_axis1", color_continuous_scale="Turbo",
                     hover_name="Name", hover_data=["iso", "family_name"],
                     projection="natural earth",
                     title=f"Languages on the map, colored by sound ({args.level})")
fig.write_html(config.RESULTS / f"worldmap_{args.level}.html")

meta.to_parquet(config.PROCESSED / f"embedding_{args.level}.parquet")