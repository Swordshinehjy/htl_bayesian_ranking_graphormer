
"""
Molecular feature extraction unit tests
"""
import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from htl_package import (
    _one_hot, _atom_features, _bond_features,
    featurize_mol, MolGraphData, ATOM_FDIM, BOND_FDIM
)


class TestOneHot(unittest.TestCase):
    """Test _one_hot function"""

    def test_one_hot_match(self):
        """Test matching case"""
        choices = ['A', 'B', 'C']
        result = _one_hot('B', choices)
        expected = [0, 1, 0, 0]  # 3 choices + 1 unknown
        self.assertEqual(result, expected)

    def test_one_hot_no_match(self):
        """Test non-matching case"""
        choices = ['A', 'B', 'C']
        result = _one_hot('D', choices)
        expected = [0, 0, 0, 1]  # 3 choices + 1 unknown
        self.assertEqual(result, expected)

    def test_one_hot_length(self):
        """Test return length"""
        choices = ['A', 'B', 'C', 'D']
        result = _one_hot('B', choices)
        self.assertEqual(len(result), len(choices) + 1)


class TestAtomFeatures(unittest.TestCase):
    """Test _atom_features function"""

    def test_atom_features_shape(self):
        """Test feature vector shape"""
        mol = Chem.MolFromSmiles('CC')
        atom = mol.GetAtomWithIdx(0)
        features = _atom_features(atom)
        self.assertEqual(len(features), ATOM_FDIM)
        self.assertEqual(features.dtype, np.float32)

    def test_atom_features_carbon(self):
        """Test carbon atom feature"""
        mol = Chem.MolFromSmiles('C')
        atom = mol.GetAtomWithIdx(0)
        features = _atom_features(atom)
        # Carbon is the first element of _ATOM_SYMBOLS
        self.assertEqual(features[0], 1)  # C one-hot
        self.assertEqual(features[1], 0)  # Not N

    def test_atom_features_oxygen(self):
        """Test oxygen atom feature"""
        mol = Chem.MolFromSmiles('O')
        atom = mol.GetAtomWithIdx(0)
        features = _atom_features(atom)
        # Oxygen is the third element of _ATOM_SYMBOLS
        self.assertEqual(features[0], 0)  # Not C
        self.assertEqual(features[1], 0)  # Not N
        self.assertEqual(features[2], 1)  # O one-hot


class TestBondFeatures(unittest.TestCase):
    """Test _bond_features function"""

    def test_bond_features_shape(self):
        """Test bond feature vector shape"""
        mol = Chem.MolFromSmiles('CC')
        bond = mol.GetBondWithIdx(0)
        features = _bond_features(bond)
        self.assertEqual(len(features), BOND_FDIM)
        self.assertEqual(features.dtype, np.float32)

    def test_bond_features_single(self):
        """Test single bond feature vector"""
        mol = Chem.MolFromSmiles('CC')
        bond = mol.GetBondWithIdx(0)
        features = _bond_features(bond)
        # Single bond is the first element of _BOND_TYPES
        self.assertEqual(features[0], 1)  # SINGLE bond one-hot
        self.assertEqual(features[1], 0)  # Not DOUBLE

    def test_bond_features_double(self):
        """Test double bond feature vector"""
        mol = Chem.MolFromSmiles('C=O')
        bond = mol.GetBondWithIdx(0)
        features = _bond_features(bond)
        # Double bond is the second element of _BOND_TYPES
        self.assertEqual(features[0], 0)  # Not SINGLE
        self.assertEqual(features[1], 1)  # DOUBLE bond one-hot


class TestFeaturizeMol(unittest.TestCase):
    """Test featurize_mol function"""

    def test_featurize_simple_mol(self):
        """Test simple molecule featureaturization"""
        mol = Chem.MolFromSmiles('CC')
        mol_data = featurize_mol(mol, max_dist=10)

        self.assertIsNotNone(mol_data)
        self.assertIsInstance(mol_data, MolGraphData)
        self.assertEqual(mol_data.n_atoms, 2)
        self.assertEqual(mol_data.atom_feats.shape, (2, ATOM_FDIM))
        self.assertEqual(mol_data.dist_matrix.shape, (2, 2))
        self.assertEqual(mol_data.edge_path_feats.shape, (2, 2, 10, BOND_FDIM))
        self.assertEqual(len(mol_data.degree), 2)

    def test_featurize_none_mol(self):
        """Test None molecule"""
        mol_data = featurize_mol(None, max_dist=10)
        self.assertIsNone(mol_data)

    def test_featurize_empty_mol(self):
        """Test empty molecule"""
        mol = Chem.MolFromSmiles('')
        mol_data = featurize_mol(mol, max_dist=10)
        self.assertIsNone(mol_data)

    def test_featurize_complex_mol(self):
        """Test complex molecule featureaturization"""
        mol = Chem.MolFromSmiles('c1ccccc1')  # Benzene ring
        mol_data = featurize_mol(mol, max_dist=10)

        self.assertIsNotNone(mol_data)
        self.assertEqual(mol_data.n_atoms, 6)
        self.assertEqual(mol_data.atom_feats.shape, (6, ATOM_FDIM))
        self.assertEqual(mol_data.dist_matrix.shape, (6, 6))
        self.assertEqual(mol_data.edge_path_feats.shape, (6, 6, 10, BOND_FDIM))

    def test_distance_matrix(self):
        """Test distance matrix"""
        mol = Chem.MolFromSmiles('CCC')  # Propane
        mol_data = featurize_mol(mol, max_dist=10)

        # Check diagonal is 0
        for i in range(mol_data.n_atoms):
            self.assertEqual(mol_data.dist_matrix[i, i], 0)

        # Check adjacent atoms distance is 1
        self.assertEqual(mol_data.dist_matrix[0, 1], 1)
        self.assertEqual(mol_data.dist_matrix[1, 2], 1)

        # Check non-adjacent atoms distance
        self.assertEqual(mol_data.dist_matrix[0, 2], 2)


class TestMolGraphData(unittest.TestCase):
    """Test MolGraphData class"""

    def test_mol_graph_data_creation(self):
        """Test MolGraphData creation"""
        mol = Chem.MolFromSmiles('CC')
        mol_data = featurize_mol(mol, max_dist=10)

        self.assertIsInstance(mol_data.atom_feats, np.ndarray)
        self.assertIsInstance(mol_data.dist_matrix, np.ndarray)
        self.assertIsInstance(mol_data.edge_path_feats, np.ndarray)
        self.assertIsInstance(mol_data.degree, np.ndarray)
        self.assertIsInstance(mol_data.n_atoms, int)

        self.assertEqual(mol_data.atom_feats.dtype, np.float32)
        self.assertEqual(mol_data.dist_matrix.dtype, np.int16)
        self.assertEqual(mol_data.edge_path_feats.dtype, np.float32)
        self.assertEqual(mol_data.degree.dtype, np.int64)


if __name__ == '__main__':
    unittest.main()
