
"""
Model architecture unit tests
"""
import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from htl_package import (
    EdgeEncoding, GraphormerLayer, GraphormerEncoder,
    HTLRankingModel, MolGraphData, MolBatch, collate_mol_graphs,
    featurize_mol, ATOM_FDIM, BOND_FDIM, EXTRA_DIM, GLOBAL_DIM, NUM_TASKS
)


class TestEdgeEncoding(unittest.TestCase):
    """Test EdgeEncoding module"""

    def setUp(self):
        self.edge_dim = 12
        self.num_heads = 4
        self.max_path_len = 10
        self.module = EdgeEncoding(self.edge_dim, self.num_heads, self.max_path_len)

    def test_edge_encoding_shape(self):
        """Test output shape"""
        batch_size = 2
        n_atoms = 5
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, self.max_path_len, self.edge_dim)

        edge_bias = self.module(edge_path_feats)
        self.assertEqual(edge_bias.shape, (batch_size, n_atoms, n_atoms, self.num_heads))

    def test_edge_encoding_forward(self):
        """Test forward propagation"""
        batch_size = 2
        n_atoms = 5
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, self.max_path_len, self.edge_dim)

        edge_bias = self.module(edge_path_feats)
        self.assertFalse(torch.isnan(edge_bias).any())
        self.assertFalse(torch.isinf(edge_bias).any())


class TestGraphormerLayer(unittest.TestCase):
    """Test GraphormerLayer module"""

    def setUp(self):
        self.hidden_size = 64
        self.num_heads = 4
        self.ffn_hidden = 256
        self.dropout = 0.1
        self.layer = GraphormerLayer(
            self.hidden_size, self.num_heads, self.ffn_hidden, self.dropout
        )

    def test_graphormer_layer_shape(self):
        """Test output shape"""
        batch_size = 2
        n_atoms = 10
        x = torch.randn(batch_size, n_atoms, self.hidden_size)
        attn_bias = torch.randn(batch_size * self.num_heads, n_atoms, n_atoms)
        key_padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        output = self.layer(x, attn_bias, key_padding_mask)
        self.assertEqual(output.shape, (batch_size, n_atoms, self.hidden_size))

    def test_graphormer_layer_forward(self):
        """Test forward propagation"""
        batch_size = 2
        n_atoms = 10
        x = torch.randn(batch_size, n_atoms, self.hidden_size)
        attn_bias = torch.randn(batch_size * self.num_heads, n_atoms, n_atoms)
        key_padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        output = self.layer(x, attn_bias, key_padding_mask)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestGraphormerEncoder(unittest.TestCase):
    """Test GraphormerEncoder module"""

    def setUp(self):
        self.hidden_size = 64
        self.depth = 2
        self.num_heads = 4
        self.dropout = 0.1
        self.max_degree = 10
        self.max_dist = 10
        self.encoder = GraphormerEncoder(
            self.hidden_size, self.depth, self.num_heads, self.dropout,
            self.max_degree, self.max_dist, aggregation="cls"
        )

    def test_encoder_shape_cls(self):
        """Test output shape for CLS aggregation"""
        batch_size = 2
        n_atoms = 10
        atom_feats = torch.randn(batch_size, n_atoms, ATOM_FDIM)
        dist_matrix = torch.randint(0, self.max_dist, (batch_size, n_atoms, n_atoms))
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, self.max_dist, BOND_FDIM)
        degree = torch.randint(0, self.max_degree, (batch_size, n_atoms))
        padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        batch = MolBatch(atom_feats, dist_matrix, edge_path_feats, degree, padding_mask, [n_atoms, n_atoms])
        output = self.encoder(batch)
        self.assertEqual(output.shape, (batch_size, self.hidden_size))

    def test_encoder_shape_mean(self):
        """Test output shape for MEAN aggregation"""
        encoder = GraphormerEncoder(
            self.hidden_size, self.depth, self.num_heads, self.dropout,
            self.max_degree, self.max_dist, aggregation="mean"
        )

        batch_size = 2
        n_atoms = 10
        atom_feats = torch.randn(batch_size, n_atoms, ATOM_FDIM)
        dist_matrix = torch.randint(0, self.max_dist, (batch_size, n_atoms, n_atoms))
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, self.max_dist, BOND_FDIM)
        degree = torch.randint(0, self.max_degree, (batch_size, n_atoms))
        padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        batch = MolBatch(atom_feats, dist_matrix, edge_path_feats, degree, padding_mask, [n_atoms, n_atoms])
        output = encoder(batch)
        self.assertEqual(output.shape, (batch_size, self.hidden_size))

    def test_encoder_forward(self):
        """Test forward propagation"""
        batch_size = 2
        n_atoms = 10
        atom_feats = torch.randn(batch_size, n_atoms, ATOM_FDIM)
        dist_matrix = torch.randint(0, self.max_dist, (batch_size, n_atoms, n_atoms))
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, self.max_dist, BOND_FDIM)
        degree = torch.randint(0, self.max_degree, (batch_size, n_atoms))
        padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        batch = MolBatch(atom_feats, dist_matrix, edge_path_feats, degree, padding_mask, [n_atoms, n_atoms])
        output = self.encoder(batch)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestHTLRankingModel(unittest.TestCase):
    """Test HTLRankingModel module"""

    def setUp(self):
        self.model = HTLRankingModel(
            hidden_size=64,
            depth=2,
            num_heads=4,
            dropout=0.1,
            ffn_hidden=128,
            extra_dim=EXTRA_DIM,
            global_dim=GLOBAL_DIM,
            num_tasks=NUM_TASKS,
            max_degree=10,
            max_dist=10
        )

    def test_model_shape(self):
        """Test output shape"""
        batch_size = 2
        n_atoms = 10

        # create two molecular graphs
        atom_feats = torch.randn(batch_size, n_atoms, ATOM_FDIM)
        dist_matrix = torch.randint(0, 10, (batch_size, n_atoms, n_atoms))
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, 10, BOND_FDIM)
        degree = torch.randint(0, 10, (batch_size, n_atoms))
        padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        mb1 = MolBatch(atom_feats, dist_matrix, edge_path_feats, degree, padding_mask, [n_atoms, n_atoms])
        mb2 = MolBatch(atom_feats.clone(), dist_matrix.clone(), edge_path_feats.clone(), 
                      degree.clone(), padding_mask.clone(), [n_atoms, n_atoms])

        ef1 = torch.randn(batch_size, EXTRA_DIM)
        ef2 = torch.randn(batch_size, EXTRA_DIM)
        gf = torch.randn(batch_size, GLOBAL_DIM)

        s1, s2 = self.model(mb1, ef1, mb2, ef2, gf)
        self.assertEqual(s1.shape, (batch_size, NUM_TASKS))
        self.assertEqual(s2.shape, (batch_size, NUM_TASKS))
    def test_model_forward(self):
        """Test forward propagation"""
        batch_size = 2
        n_atoms = 10

        # create two molecular graphs
        atom_feats = torch.randn(batch_size, n_atoms, ATOM_FDIM)
        dist_matrix = torch.randint(0, 10, (batch_size, n_atoms, n_atoms))
        edge_path_feats = torch.randn(batch_size, n_atoms, n_atoms, 10, BOND_FDIM)
        degree = torch.randint(0, 10, (batch_size, n_atoms))
        padding_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)

        mb1 = MolBatch(atom_feats, dist_matrix, edge_path_feats, degree, padding_mask, [n_atoms, n_atoms])
        mb2 = MolBatch(atom_feats.clone(), dist_matrix.clone(), edge_path_feats.clone(), 
                      degree.clone(), padding_mask.clone(), [n_atoms, n_atoms])

        ef1 = torch.randn(batch_size, EXTRA_DIM)
        ef2 = torch.randn(batch_size, EXTRA_DIM)
        gf = torch.randn(batch_size, GLOBAL_DIM)

        s1, s2 = self.model(mb1, ef1, mb2, ef2, gf)
        self.assertFalse(torch.isnan(s1).any())
        self.assertFalse(torch.isnan(s2).any())
        self.assertFalse(torch.isinf(s1).any())
        self.assertFalse(torch.isinf(s2).any())

    def test_model_encode(self):
        """Test encode method with real molecular data"""
        mols = [Chem.MolFromSmiles('CC'), Chem.MolFromSmiles('c1ccccc1')]
        mol_data_list = [featurize_mol(mol, max_dist=10) for mol in mols]
        batch = collate_mol_graphs(mol_data_list, max_dist=10)

        ef = torch.randn(2, EXTRA_DIM)
        gf = torch.randn(2, GLOBAL_DIM)

        scores = self.model.encode(batch, ef, gf)
        self.assertEqual(scores.shape, (2, NUM_TASKS))
        self.assertFalse(torch.isnan(scores).any())
        self.assertFalse(torch.isinf(scores).any())


class TestCollateMolGraphs(unittest.TestCase):
    """Test collate_mol_graphs function"""

    def test_collate_single_graph(self):
        """Test collate for single graph batch"""
        mol = Chem.MolFromSmiles('CC')
        mol_data = featurize_mol(mol, max_dist=10)

        batch = collate_mol_graphs([mol_data], max_dist=10)

        self.assertIsInstance(batch, MolBatch)
        self.assertEqual(batch.atom_feats.shape[0], 1)  # batch size
        self.assertEqual(batch.n_atoms[0], mol_data.n_atoms)

    def test_collate_multiple_graphs(self):
        """Test collate for multiple graph batch"""
        mols = [Chem.MolFromSmiles('CC'), Chem.MolFromSmiles('c1ccccc1')]
        mol_data_list = [featurize_mol(mol, max_dist=10) for mol in mols]

        batch = collate_mol_graphs(mol_data_list, max_dist=10)

        self.assertIsInstance(batch, MolBatch)
        self.assertEqual(batch.atom_feats.shape[0], 2)  # batch size
        max_n_atoms = max(g.n_atoms for g in mol_data_list)
        self.assertEqual(batch.atom_feats.shape[1], max_n_atoms)

    def test_collate_padding(self):
        """Test padding mask creation"""
        mols = [Chem.MolFromSmiles('CC'), Chem.MolFromSmiles('c1ccccc1')]
        mol_data_list = [featurize_mol(mol, max_dist=10) for mol in mols]

        batch = collate_mol_graphs(mol_data_list, max_dist=10)

        # check padding mask
        self.assertEqual(batch.padding_mask.shape, (2, max(g.n_atoms for g in mol_data_list)))
        # the first molecule should have some padding
        self.assertTrue(batch.padding_mask[0, mol_data_list[0].n_atoms:].all())


if __name__ == '__main__':
    unittest.main()
