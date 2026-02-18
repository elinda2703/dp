# 1. Define the manual mapping
# Key = Index (0..24)
# Value = Region Letter

regions_dict = {}

regions_dict['reg25'] = {
    # --- Region A: The Center Cross (Exactly 5) ---
    # Center (12) + Up(17), Down(7), Left(11), Right(13)
    12: 'A', 
    17: 'A', 7: 'A', 11: 'A', 13: 'A',

    # --- Region B: Bottom-Left (4 corner + 1 bottom-middle) ---
    # The 2x2 corner (0,1,5,6) + the orphan at the bottom (2)
    0: 'B', 1: 'B', 5: 'B', 6: 'B', 
    2: 'B',

    # --- Region C: Bottom-Right (4 corner + 1 right-middle) ---
    # The 2x2 corner (3,4,8,9) + the orphan at the right (14)
    3: 'C', 4: 'C', 8: 'C', 9: 'C', 
    14: 'C',

    # --- Region D: Top-Right (4 corner + 1 top-middle) ---
    # The 2x2 corner (18,19,23,24) + the orphan at the top (22)
    18: 'D', 19: 'D', 23: 'D', 24: 'D', 
    22: 'D',

    # --- Region E: Top-Left (4 corner + 1 left-middle) ---
    # The 2x2 corner (15,16,20,21) + the orphan at the left (10)
    15: 'E', 16: 'E', 20: 'E', 21: 'E', 
    10: 'E'
}

regions_dict['reg100'] = {
    # --- Region A: The Center Cross ---
    # The Core Intersection (44,45,54,55) + Arms extending out
    # Vertical Arm (Cols 4-5)
    24: 'A', 25: 'A', 34: 'A', 35: 'A',
    44: 'A', 45: 'A', 54: 'A', 55: 'A',
    64: 'A', 65: 'A', 74: 'A', 75: 'A',
    # Horizontal Arm (Cols 2-3 Left, Cols 6-7 Right)
    42: 'A', 43: 'A', 52: 'A', 53: 'A',
    46: 'A', 47: 'A', 56: 'A', 57: 'A',


    # --- Region B: Bottom-Left (Blue) ---
    # The 4x4 Corner (Rows 0-3, Cols 0-3)
    0: 'B', 1: 'B', 2: 'B', 3: 'B',
    10: 'B', 11: 'B', 12: 'B', 13: 'B',
    20: 'B', 21: 'B', 22: 'B', 23: 'B',
    30: 'B', 31: 'B', 32: 'B', 33: 'B',
    # Extension: The 2x2 orphan block at the BOTTOM (Rows 0-1, Cols 4-5)
    4: 'B', 5: 'B', 
    14: 'B', 15: 'B',


    # --- Region C: Bottom-Right (Purple) ---
    # The 4x4 Corner (Rows 0-3, Cols 6-9)
    6: 'C', 7: 'C', 8: 'C', 9: 'C',
    16: 'C', 17: 'C', 18: 'C', 19: 'C',
    26: 'C', 27: 'C', 28: 'C', 29: 'C',
    36: 'C', 37: 'C', 38: 'C', 39: 'C',
    # Extension: The 2x2 orphan block at the RIGHT (Rows 4-5, Cols 8-9)
    48: 'C', 49: 'C',
    58: 'C', 59: 'C',


    # --- Region D: Top-Right (Gray) ---
    # The 4x4 Corner (Rows 6-9, Cols 6-9)
    66: 'D', 67: 'D', 68: 'D', 69: 'D',
    76: 'D', 77: 'D', 78: 'D', 79: 'D',
    86: 'D', 87: 'D', 88: 'D', 89: 'D',
    96: 'D', 97: 'D', 98: 'D', 99: 'D',
    # Extension: The 2x2 orphan block at the TOP (Rows 8-9, Cols 4-5)
    84: 'D', 85: 'D',
    94: 'D', 95: 'D',


    # --- Region E: Top-Left (Orange) ---
    # The 4x4 Corner (Rows 6-9, Cols 0-3)
    60: 'E', 61: 'E', 62: 'E', 63: 'E',
    70: 'E', 71: 'E', 72: 'E', 73: 'E',
    80: 'E', 81: 'E', 82: 'E', 83: 'E',
    90: 'E', 91: 'E', 92: 'E', 93: 'E',
    # Extension: The 2x2 orphan block at the LEFT (Rows 4-5, Cols 0-1)
    40: 'E', 41: 'E',
    50: 'E', 51: 'E'
}


regions_dict['reg400'] = {}

# We define a helper to add a block of IDs to a specific region
# This saves us from typing 400 numbers manually
def add_block(rows, cols, region_char):
    for r in rows:
        for c in cols:
            # Formula for ID in a 20x20 grid: Row * 20 + Col
            regions_dict['reg400'][r * 20 + c] = region_char


# --- Region B: Bottom-Left (Blue) ---
# Base: The big 8x8 corner at Bottom-Left
add_block(rows=range(0, 8), cols=range(0, 8), region_char='B')
# Extension: The 4x4 "Tip" taken from the Bottom Center
add_block(rows=range(0, 4), cols=range(8, 12), region_char='B')


# --- Region C: Bottom-Right (Purple) ---
# Base: The big 8x8 corner at Bottom-Right
add_block(rows=range(0, 8), cols=range(12, 20), region_char='C')
# Extension: The 4x4 "Tip" taken from the Right Center
add_block(rows=range(8, 12), cols=range(16, 20), region_char='C')


# --- Region D: Top-Right (Gray) ---
# Base: The big 8x8 corner at Top-Right
add_block(rows=range(12, 20), cols=range(12, 20), region_char='D')
# Extension: The 4x4 "Tip" taken from the Top Center
add_block(rows=range(16, 20), cols=range(8, 12), region_char='D')


# --- Region E: Top-Left (Orange) ---
# Base: The big 8x8 corner at Top-Left
add_block(rows=range(12, 20), cols=range(0, 8), region_char='E')
# Extension: The 4x4 "Tip" taken from the Left Center
add_block(rows=range(8, 12), cols=range(0, 4), region_char='E')


# --- Region A: The Center Cross (Red) ---
# This fills in the gaps (The Center Hub + Inner Arms)
# We fill the 'middle belt' of the grid, skipping what we already assigned to B, C, D, E.
# (Rows 8-11 and Cols 8-11, plus the connecting arms)

# Center Core (4x4)
add_block(rows=range(8, 12), cols=range(8, 12), region_char='A')

# Remaining Arms (The parts not eaten by the corners)
add_block(rows=range(4, 8),   cols=range(8, 12), region_char='A') # Bottom Arm Stub
add_block(rows=range(8, 12),  cols=range(12, 16), region_char='A') # Right Arm Stub
add_block(rows=range(12, 16), cols=range(8, 12), region_char='A') # Top Arm Stub
add_block(rows=range(8, 12),  cols=range(4, 8),   region_char='A') # Left Arm Stub

#--------------------------------------------------------------------------------------
"""
regions_dict['hex25'] = {
    # --- Region A: The Center Cross (Exactly 5) ---
    # Center (12) + Up(17), Down(7), Left(11), Right(13)
    12: 'A', 
    17: 'A', 10: 'A', 5: 'A', 13: 'A',

    # --- Region B: Bottom-Left (4 corner + 1 bottom-middle) ---
    # The 2x2 corner (0,1,5,6) + the orphan at the bottom (2)
    0: 'B', 1: 'B', 3: 'B', 4: 'B', 
    2: 'B',

    # --- Region C: Bottom-Right (4 corner + 1 right-middle) ---
    # The 2x2 corner (3,4,8,9) + the orphan at the right (14)
    6: 'C', 7: 'C', 8: 'C', 9: 'C', 
    14: 'C',

    # --- Region D: Top-Right (4 corner + 1 top-middle) ---
    # The 2x2 corner (18,19,23,24) + the orphan at the top (22)
    18: 'D', 19: 'D', 23: 'D', 24: 'D', 
    22: 'D',

    # --- Region E: Top-Left (4 corner + 1 left-middle) ---
    # The 2x2 corner (15,16,20,21) + the orphan at the left (10)
    15: 'E', 16: 'E', 20: 'E', 21: 'E', 
    11: 'E'
}


# ==========================================
# PART 1: HEX 100 (10x10 Grid)
# ==========================================
regions_dict['hex100'] = {}

def add_block_100(rows, cols, region_char):
    for r in rows:
        for c in cols:
            # Assumes standard ID = Row * 10 + Col
            regions_dict['hex100'][r * 10 + c] = region_char

# --- Region B: Bottom-Left (Blue) ---
# Corner (4x4) + Extension Right (along bottom)
add_block_100(rows=range(0, 4), cols=range(0, 4), region_char='B') 
add_block_100(rows=range(0, 2), cols=range(4, 6), region_char='B') # The "Orphan"

# --- Region C: Bottom-Right (Purple) ---
# Corner (4x4) + Extension Up (along right)
add_block_100(rows=range(0, 4), cols=range(6, 10), region_char='C')
add_block_100(rows=range(4, 6), cols=range(8, 10), region_char='C') # The "Orphan"

# --- Region D: Top-Right (Gray) ---
# Corner (4x4) + Extension Left (along top)
add_block_100(rows=range(6, 10), cols=range(6, 10), region_char='D')
add_block_100(rows=range(8, 10), cols=range(4, 6), region_char='D') # The "Orphan"

# --- Region E: Top-Left (Orange) ---
# Corner (4x4) + Extension Down (along left)
add_block_100(rows=range(6, 10), cols=range(0, 4), region_char='E')
add_block_100(rows=range(4, 6), cols=range(0, 2), region_char='E') # The "Orphan"

# --- Region A: Center Cross (Red) ---
# Fills the remaining gaps (Center 2x2 + Arms)
# We simply loop over the "Middle Belt" indices not yet assigned
for r in range(2, 8):
    for c in range(2, 8):
        idx = r * 10 + c
        if idx not in regions_dict['hex100']:
            regions_dict['hex100'][idx] = 'A'


# ==========================================
# PART 2: HEX 400 (20x20 Grid)
# ==========================================
regions_dict['hex400'] = {}

def add_block_400(rows, cols, region_char):
    for r in rows:
        for c in cols:
            # Assumes standard ID = Row * 20 + Col
            regions_dict['hex400'][r * 20 + c] = region_char

# --- Region B: Bottom-Left ---
# Corner (8x8) + Extension Right (4x4 block)
add_block_400(rows=range(0, 8), cols=range(0, 8), region_char='B')
add_block_400(rows=range(0, 4), cols=range(8, 12), region_char='B')

# --- Region C: Bottom-Right ---
# Corner (8x8) + Extension Up (4x4 block)
add_block_400(rows=range(0, 8), cols=range(12, 20), region_char='C')
add_block_400(rows=range(8, 12), cols=range(16, 20), region_char='C')

# --- Region D: Top-Right ---
# Corner (8x8) + Extension Left (4x4 block)
add_block_400(rows=range(12, 20), cols=range(12, 20), region_char='D')
add_block_400(rows=range(16, 20), cols=range(8, 12), region_char='D')

# --- Region E: Top-Left ---
# Corner (8x8) + Extension Down (4x4 block)
add_block_400(rows=range(12, 20), cols=range(0, 8), region_char='E')
add_block_400(rows=range(8, 12), cols=range(0, 4), region_char='E')

# --- Region A: Center Cross ---
# Fill the rest
for r in range(4, 16):
    for c in range(4, 16):
        idx = r * 20 + c
        if idx not in regions_dict['hex400']:
            regions_dict['hex400'][idx] = 'A'
            
#-------------------------------------------------------------- 

pent25 = {
    # --- Region A: The Center Hub (4 cells) ---
    # The tight 2x2 cluster in the middle
    6: 'A', 9: 'A', 15: 'A', 16: 'A', 

    # --- Region B: Bottom-Left (6 cells) ---
    # The Bottom-Left Corner + The Bottom "Orphan" (4)
    0: 'B', 2: 'B',       # Col 0 (Bottom)
    5: 'B', 8: 'B',     # Col 1 (Bottom)
    4: 'B',               # The Bottom Hook (Col 2)

    # --- Region C: Bottom-Right (6 cells) ---
    # The Bottom-Right Corner + The Right "Orphan" (14)
    13: 'C',              # Col 3 (Bottom)
    12: 'C', 17: 'C',     # Col 4 (Bottom)
    20: 'C', 22: 'C',     # Col 5 (Bottom)             

    # --- Region D: Top-Right (5 cells) ---
    # The Top-Right Corner + The Top "Orphan" (11)
    19: 'D',              # Col 4 (Top)
    23: 'D', 21: 'D',     # Col 5 (Top)
    18: 'D',              # Col 3 (Top)
    14: 'D',              # The Top Hook (Col 2)

    # --- Region E: Top-Left (3 cells) ---
    # The Top-Left Corner + The Left "Orphan" (10)
    3: 'E', 1: 'E',       # Col 0 (Top)
    10: 'E', 7: 'E', 11: 'E'               # The Left Hook (Col 1)
}"""           