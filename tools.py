from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import spsolve
import geopandas as gpd
#import pandas as pd
import numpy as np
#from libpysal.weights import Queen, Rook
import matplotlib.pyplot as plt
from esda.moran import Moran
from shapely.geometry import Polygon
#import copy
import seaborn as sns
from libpysal import graph
from scipy.sparse import spmatrix, triu, diags, coo_array

def generate_square_lattice(l):    
    # Get points in a grid
    l = np.arange(l)
    xs, ys = np.meshgrid(l, l)
    # Set up store
    polys = []
    # Generate polygons
    for x, y in zip(xs.flatten(), ys.flatten()):
        poly = Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
        polys.append(poly)
    # Convert to GeoSeries
    polys = gpd.GeoSeries(polys)
    gdf = gpd.GeoDataFrame(
        {
            "geometry": polys,
            "id": ["P-%s" % str(i).zfill(2) for i in range(len(polys))],
        }
    )
    
    return gdf

def build_g(gdf):
    # 2. Build spatial weights (Rook contiguity)
    g = graph.Graph.build_contiguity(gdf,rook=True)
    g = g.transform("R")
    return g

def remove_random_edges(W_sparse, proportion: float):
    """
    Randomly remove a proportion of off-diagonal edges 
    from a symmetric SciPy sparse adjacency matrix.
    """
    if not 0.0 <= proportion <= 1.0:
        raise ValueError("Proportion must be between 0 and 1")

    if proportion == 0.0:
        return W_sparse.copy()

    # Keep diagonal
    diag = diags(W_sparse.diagonal(), format="coo")

    # Upper triangle (no diagonal)
    W_upper = triu(W_sparse, k=1).tocoo()
    n_edges = W_upper.nnz
    if n_edges == 0:
        return W_sparse.copy()

    # Randomly keep a subset of edges
    n_keep = int(round(n_edges * (1 - proportion)))
    keep_idx = np.random.choice(np.arange(n_edges), size=n_keep, replace=False)

    rows = W_upper.row[keep_idx]
    cols = W_upper.col[keep_idx]
    data = W_upper.data[keep_idx]

    # Symmetrize
    rows_sym = np.concatenate([rows, cols])
    cols_sym = np.concatenate([cols, rows])
    data_sym = np.concatenate([data, data])

    # Build new symmetric sparse matrix
    W_new = coo_array((data_sym, (rows_sym, cols_sym)), shape=W_sparse.shape)
    W_new = (W_new + diag)
    return W_new