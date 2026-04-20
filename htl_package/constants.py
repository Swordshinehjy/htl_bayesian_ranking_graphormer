import logging
import warnings

import torch
from rdkit import Chem

warnings.filterwarnings("ignore")

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")
