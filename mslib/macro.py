import numpy as np

ALL_OPERATIONS = ['RS', 'Mill', 'Drill', 'Slant', 'EOS', 'FP']
RS_IDX = ALL_OPERATIONS.index('RS') # Raw Stock
MILL_IDX = ALL_OPERATIONS.index('Mill')
DRILL_IDX = ALL_OPERATIONS.index('Drill')
SLANT_IDX = ALL_OPERATIONS.index('Slant')
EOS_IDX = ALL_OPERATIONS.index('EOS') # End of Sequence
FP_IDX = ALL_OPERATIONS.index('FP') # Final Part

N_OPERATIONS = len(ALL_OPERATIONS) - 1 # the number of operations except for 'FP'

MAX_N_MILL = 4
MAX_N_DRILL = 5
MAX_N_SLANT = 1
MAX_TOTAL_LEN = MAX_N_DRILL + MAX_N_MILL + MAX_N_SLANT + 2 # maximum process planning sequence length (5 DRILL + 4 MILL + 1 SLANT + 1 RS + 1 EOS)

VOX_DIM = 128 # dimension of the voxel grid