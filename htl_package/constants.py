import logging
import warnings

import torch
from rdkit import Chem

# Silence known third-party noise without hiding warnings from this package
# or other dependencies.
warnings.filterwarnings("ignore", category=FutureWarning,
                        module=r"sklearn|pandas")
warnings.filterwarnings("ignore", category=UserWarning, module=r"rdkit")

EXTRA_COLS = [
    "Alkyl_{s}",
    "TailSym_{s}",
    "TailPlanarity_{s}",
    "NumHAcceptors_{s}",
    "NumHDonors_{s}",
    "TPSA_{s}",
    "MolLogP_{s}",
    "HOMO_{s}",
    "dipole_{s}",
    "MPI_{s}",
    "surface_min_{s}",
    "surface_max_{s}",
    "PSA_{s}",
]
EXTRA_DIM = len(EXTRA_COLS)

GLOBAL_COLS = ["MO_ITO"]
GLOBAL_DIM = len(GLOBAL_COLS)
TASK_NAMES = ["PCE"]
NUM_TASKS = len(TASK_NAMES)

_ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'Se', 'Te',
    'As', 'Sn', 'Ge',
]

_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

_STEREO_TYPES = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]

# Package-level logger with no forced handlers: importing this package must
# not touch the environment. Entry points (e.g. htl_ranking_graphormer.py)
# configure logging themselves; records propagate to the root logger.
logger = logging.getLogger("htl_package")
logger.addHandler(logging.NullHandler())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
