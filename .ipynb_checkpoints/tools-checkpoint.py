from scipy.sparse import identity, csr_matrix
from scipy.sparse.linalg import spsolve
import geopandas as gpd
import pandas as pd
import math
#import pandas as pd
import numpy as np
#from libpysal.weights import Queen, Rook
import matplotlib.pyplot as plt
from esda.moran import Moran
from shapely.geometry import LineString, Polygon, box
from scipy.spatial import Voronoi
from shapely.ops import polygonize, unary_union
import seaborn as sns
from libpysal import graph
from scipy.sparse import spmatrix, triu, diags, coo_array
#from hilbertcurve.hilbertcurve import HilbertCurve

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
            "geometry": polys},
        index=range(len(polys))      )
    
    return gdf


def generate_hex_lattice(cols, rows, size=1):
    """
    Generates a hex grid by first creating a wireframe mesh of lines,
    then 'polygonizing' the closed loops. This guarantees perfect topology.
    """
    # 1. Math Setup (Pointy Top Hexagons)
    # -----------------------------------
    # w = width (x-axis distance)
    # h = height (y-axis distance)
    w = np.sqrt(3) * size
    h = 2 * size
    
    # Vertical spacing between rows is 3/4 of height
    vert_step = 1.5 * size
    # Horizontal spacing is full width
    horiz_step = w
    
    lines = []

    # 2. Generate the "Wireframe" (The Lines)
    # ---------------------------------------
    # Instead of making full polygons, we calculate the 6 segments for every logical hex
    # and dump them into a bucket. 
    # Note: This produces duplicate lines for shared edges, but `unary_union`
    # will fix that in the next step automatically.
    
    for row in range(rows):
        for col in range(cols):
            # Calculate center of this logical hex
            x_offset = (w / 2) if (row % 2 == 1) else 0
            cx = (col * horiz_step) + x_offset
            cy = row * vert_step
            
            # Generate the 6 vertices
            angles = np.radians([30, 90, 150, 210, 270, 330])
            vx = cx + size * np.cos(angles)
            vy = cy + size * np.sin(angles)
            
            # Create 6 LineStrings (edges) for this hex location
            # (Connecting v0-v1, v1-v2, ... v5-v0)
            for i in range(6):
                start_pt = (vx[i], vy[i])
                end_pt = (vx[(i+1)%6], vy[(i+1)%6])
                lines.append(LineString([start_pt, end_pt]))

    # 3. Create the "Mesh" (The Topology Magic)
    # -----------------------------------------
    # unary_union does two critical things:
    # A. It merges duplicate lines (shared edges become one geometry).
    # B. It "nodes" intersections (ensures lines connect exactly at endpoints).
    mesh = unary_union(lines)
    
    # 4. Polygonize
    # -------------
    # Finds all closed loops in the linestring mesh and turns them into Polygons.
    polys = list(polygonize(mesh))
    
    # 5. Create GeoDataFrame
    # ----------------------
    gdf = gpd.GeoDataFrame(
        {"geometry": polys},
        # Create an ID based on order (or calculate centroid to assign row/col ID)
        index=range(len(polys))
    )
    
    return gdf


def generate_pent_lattice(width, height, angle_deg=15, scale=1.0):
    """
    Generates a contiguous Pentagonal Lattice (Cairo Tiling) using Voronoi.
    
    How it works:
    1. We create a grid of 'seed points'.
    2. At every grid intersection, we place 4 points arranged in a 
       slightly rotated square.
    3. The Voronoi diagram of these specific points forms a pentagonal tiling.
    """
    
    # 1. Generate Seed Points (The "Snub Square" Vertices)
    # ----------------------------------------------------
    seeds = []
    
    # Pre-calculate rotation math
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # Radius of the mini-square of dots at each grid node
    # 0.3 is a "magic number" that ensures good pentagon shapes relative to grid size 1.0
    r = 0.3 * scale 
    
    # Create a buffer of extra points around the edges to ensure 
    # the border cells are generated correctly before we clip them.
    pad = 1
    
    for x in range(-pad, width + pad):
        for y in range(-pad, height + pad):
            # Center of this grid block
            cx, cy = x * scale, y * scale
            
            # The 4 vertices of the tilted square around this center
            # relative offsets: (r, r), (-r, r), (-r, -r), (r, -r)
            # rotated by theta
            
            offsets = [
                (r, r), 
                (-r, r), 
                (-r, -r), 
                (r, -r)
            ]
            
            for ox, oy in offsets:
                # Rotate
                rox = ox * c - oy * s
                roy = ox * s + oy * c
                
                seeds.append([cx + rox, cy + roy])

    seeds = np.array(seeds)

    # 2. Compute Voronoi Diagram
    # --------------------------
    vor = Voronoi(seeds)

    # 3. Process Regions into Polygons
    # --------------------------------
    polys = []
    
    for region_index in vor.point_region:
        region = vor.regions[region_index]
        
        # -1 indicates an infinite region (at the very edge of infinity), skip it
        if -1 in region or len(region) == 0:
            continue
            
        # Get vertices for this region
        vertices = vor.vertices[region]
        polys.append(Polygon(vertices))

    # 4. Clip to Bounding Box
    # -----------------------
    # Since Voronoi generates ragged edges, we clip it cleanly to our desired dimensions.
    # We construct the box slightly inside the generated area to avoid edge artifacts.
    
    # Create the valid area box
    clip_box = box(0, 0, (width-1) * scale, (height-1) * scale)
    
    # Convert to GeoSeries
    gs = gpd.GeoSeries(polys)
    
    # Clip! This cuts off the messy outer edges
    gs_clipped = gs.intersection(clip_box)
    
    # Filter out empty or tiny slivers from the clip
    gs_clipped = gs_clipped[~gs_clipped.is_empty & (gs_clipped.area > (0.1 * scale**2))]

    # 5. Final DataFrame
    gdf = gpd.GeoDataFrame(
        {
            "geometry": gs_clipped
        }
        
    )
    
    gdf.reset_index(drop=True, inplace=True)
    
    return gdf


def build_g_rook(gdf):
    # 2. Build spatial weights (Rook contiguity)
    g = graph.Graph.build_contiguity(gdf,rook=True)
    #g = g.transform("R")
    return g

def build_g_borders(gdf, g_rook):
    g_block = graph.Graph.build_block_contiguity(gdf['region'])
    g_borders = g_rook.difference(g_block)
    g_remaining = g_rook.difference(g_borders)
    #g_borders = g_borders.transform('R')
    #g_remaining = g_remaining.transform('R')
    
    return g_borders,g_remaining

def build_g_region(gdf,g_rook,location):
    if location == 'center':
        region_ids = gdf[gdf['region'] == 'A'].index
    else:
        region_ids = gdf[gdf['region'] == 'D'].index
        
    edges_df = g_rook.adjacency.reset_index()
    # 3. Create a filter: Keep rows where EITHER 'focal' OR 'neighbor' is in Region A
    mask = edges_df['focal'].isin(region_ids) | edges_df['neighbor'].isin(region_ids)

    # 4. Apply the filter to get your final list of edges
    edges_region = edges_df[mask]
    g_region = graph.Graph.from_adjacency(edges_region,'focal','neighbor','weight')
    g_remaining = g_remaining = g_rook.difference(g_region)
    #g_region = g_region.transform('R')
    #g_remaining = g_remaining.transform('R')
    
    return g_region,g_remaining
    
    
def create_corrupted_graphs(g_subset,n_corrupt,perc_missing,shape,size,corruption_method,remains=None):
    w_correct=g_subset.to_W()
    w_correct_sparse = w_correct.sparse
    for c_run in range(n_corrupt):
        for p in perc_missing:
            w_new = remove_random_edges(w_correct_sparse, p/100)
            
            g_cor = graph.Graph.from_sparse(w_new, ids=g_subset.unique_ids)
            
            
            
            if corruption_method == 'random':
                g_cor.to_parquet(f"graphs/{shape}/size_{size}/{corruption_method}/g_{int(p)}_{c_run}.parquet")
                
            else:
                g_union = g_cor.union(remains).transform('R')
                g_union.to_parquet(f"graphs/{shape}/size_{size}/{corruption_method}/g_{int(p)}_{c_run}.parquet")
            #return g_cor  

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
    
def simulate_autocorrelated_data(gdf, W, rhos, n_runs):    
    # 3. Simulate spatially autocorrelated values
    
    n = len(gdf)
    I = identity(n)

    for rho in rhos:
        for run in range(n_runs):
            epsilon = np.random.normal(0, 1, n)
            y = spsolve(I - rho * W, epsilon)
            col_name = f"rho_{rho:.1f}_run_{run}"
            gdf[col_name] = y
        #print(rho)
        
"""       
def simulate_hilbert_data(gdf,l):
    xs, ys = np.meshgrid(np.arange(l), np.arange(l))
    coords = np.column_stack((xs.flatten(), ys.flatten()))

    p = int(np.log2(l))
    hc = HilbertCurve(p, 2)

    # Compute Hilbert indices one by one
    indices = [hc.distance_from_point([x, y]) for x, y in coords]

    # Reshape if you want 2D
    hilbert_array = np.array(indices).reshape(l, l)

    gdf["hilbert_index"] = indices""" 


def plot_lattice(gdf):    
    # Plot grid geotable
    ax = gdf.plot(facecolor="w", edgecolor="k")
    
    # Remove axes
    ax.set_axis_off()
    plt.show()
    
def plot_graph(gdf, g):
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(7,7))

    # Plot the base layer (gdf)
    # 'edgecolor' handles the black edges, 'facecolor' can be set to 'none' or a light color
    gdf.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)

    # Overlay the target layer (g)
    # We use 'red' for the color to make it stand out
    g.plot(gdf,ax=ax, color='red')

    # Final touches
    ax.axis('off')  # Optional: removes the lat/long axis box

    plt.show()
    
def reindex(gdf, row_size):
    """
    Slices the bottom 'row_size' polygons using their bottom-most point (miny),
    sorts them horizontally using their left-most point (minx), and enumerates.
    """
    # 1. Make a working copy
    #gdf = gdf.copy()
    
    # 2. Extract the bounding box limits for all polygons at once
    # .bounds returns a dataframe with ['minx', 'miny', 'maxx', 'maxy']
    bounds = gdf.bounds
    
    # 'miny' is the absolute lowest point; 'minx' is the absolute furthest left
    gdf['bottom_y'] = bounds['miny']
    gdf['left_x'] = bounds['minx']
    
        # 1. Fill the column with pandas' official "missing" value
    gdf['id'] = pd.NA

    # 2. Tell pandas this column is specifically for integers that might have missing values
    gdf['id'] = gdf['id'].astype('Int64')
    
    # Sort remaining polygons by their absolute lowest point (bottom to top)
    gdf.sort_values(by='bottom_y', ascending=True, inplace=True)
    
    # Set a running counter for the IDs
    current_id = 0
    
    # Loop through the dataframe in chunks of 'row_size' (e.g., 20 at a time)
    for i in range(0, len(gdf), row_size):
        
        # A. Grab the next chunk of polygons (e.g., positions 0-19, then 20-39)
        chunk = gdf.iloc[i : i + row_size]
        
        # B. Sort just this chunk from left to right
        chunk_sorted = chunk.sort_values(by='left_x', ascending=True)
        
        # C. Generate the list of IDs for this chunk (e.g., 0 to 19)
        # Using len() ensures we don't break if the very top row has fewer than 20 polygons
        new_ids = range(current_id, current_id + len(chunk_sorted))
        
        # D. Assign these new IDs back to the main DataFrame safely using .loc
        # We match them using the index labels of the sorted chunk
        gdf.loc[chunk_sorted.index, 'id'] = new_ids
        
        # E. Update the counter for the next loop (so the next row starts at 20)
        current_id += len(chunk_sorted)

    # 5. Clean up: Sort the final dataset by the new 'id' and reset the index
    # 4. Clean up IN-PLACE
    gdf.sort_values(by='id', inplace=True)
    gdf.reset_index(drop=True, inplace=True)
    gdf.drop(columns=['bottom_y', 'left_x'], inplace=True)
    
    #return gdf

def extract_subset(master_gdf, subset_cols, subset_rows, total_cols=20, total_rows=20):
    """
    Extracts a bottom-left subset of a grid geometrically using cell centroids.
    
    Parameters:
    - master_gdf: The main GeoDataFrame (e.g., pent400).
    - subset_cols: How many columns wide the subset should be.
    - subset_rows: How many rows high the subset should be.
    - total_cols: Total columns in the master grid (default 20).
    - total_rows: Total rows in the master grid (default 20).
    """
    # 1. Get the total bounding box of the master grid
    minx, miny, maxx, maxy = master_gdf.total_bounds
    
    # 2. Calculate the spatial size of a single logical step
    x_step = (maxx - minx) / total_cols
    y_step = (maxy - miny) / total_rows
    
    # 3. Define the geographic cutoff (adding a tiny buffer to handle floating-point math safely)
    cutoff_x = minx + (x_step * subset_cols) 
    cutoff_y = miny + (y_step * subset_rows) 
    
    # 4. Filter: Keep cells whose centroids fall strictly inside the cutoffs
    # We use centroids so we don't accidentally grab cells that just bleed over the line
    mask = (master_gdf.centroid.x < cutoff_x) & (master_gdf.centroid.y < cutoff_y)
    
    subset_gdf = master_gdf[mask].copy()
    
    return subset_gdf  