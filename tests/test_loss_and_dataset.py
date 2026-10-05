
"""
Loss function and dataset unit tests
"""
import unittest
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from htl_package import (
    BayesianRankingLoss, PairDataset, ListDataset,
    CachedPairDataset, DynamicPairDataset, collate_mol_graphs,
    featurize_mol, NUM_TASKS, TASK_NAMES, EXTRA_COLS
)


class TestBayesianRankingLoss(unittest.TestCase):
    """Test BayesianRankingLoss loss function"""

    def setUp(self):
        self.loss_fn = BayesianRankingLoss(
            rank_weight=0.6,
            reg_weight=0.4,
        )

    def test_loss_shape(self):
        """Test loss output shape"""
        batch_size = 4
        s1 = torch.randn(batch_size, NUM_TASKS)
        s2 = torch.randn(batch_size, NUM_TASKS)
        y1 = torch.randn(batch_size, NUM_TASKS)
        y2 = torch.randn(batch_size, NUM_TASKS)

        loss, log = self.loss_fn(s1, s2, y1, y2)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertIsInstance(log, dict)

    def test_loss_non_negative(self):
        """Test loss is non-negative"""
        batch_size = 4
        s1 = torch.randn(batch_size, NUM_TASKS)
        s2 = torch.randn(batch_size, NUM_TASKS)
        y1 = torch.randn(batch_size, NUM_TASKS)
        y2 = torch.randn(batch_size, NUM_TASKS)

        loss, _ = self.loss_fn(s1, s2, y1, y2)
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_loss_log_keys(self):
        """Test log dictionary contains correct keys"""
        batch_size = 4
        s1 = torch.randn(batch_size, NUM_TASKS)
        s2 = torch.randn(batch_size, NUM_TASKS)
        y1 = torch.randn(batch_size, NUM_TASKS)
        y2 = torch.randn(batch_size, NUM_TASKS)

        _, log = self.loss_fn(s1, s2, y1, y2)

        for task_name in TASK_NAMES:
            self.assertIn(f"{task_name}_rank", log)
            self.assertIn(f"{task_name}_reg", log)
        self.assertIn("total", log)


class TestPairDataset(unittest.TestCase):
    """Test PairDataset dataset"""

    def setUp(self):
        # Use the first two rows of htl-data-combinations.csv
        df_full = pd.read_csv('htl-data-combinations.csv')
        self.df = df_full.head(2).copy()
        # Convert SMILES to Mol objects
        self.df['mol_1'] = self.df['SMILES_1'].apply(Chem.MolFromSmiles)
        self.df['mol_2'] = self.df['SMILES_2'].apply(Chem.MolFromSmiles)
        self.max_dist = 10

    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = PairDataset(self.df, fit_scaler=True, max_dist=self.max_dist)

        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), len(self.df))
        self.assertIsNotNone(dataset.scaler)

    def test_dataset_getitem(self):
        """Test dataset sample retrieval"""
        dataset = PairDataset(self.df, fit_scaler=True, max_dist=self.max_dist)

        sample = dataset[0]

        self.assertEqual(len(sample), 7)  # (g1, g2, ef1, ef2, gf, y1, y2)
        g1, g2, ef1, ef2, gf, y1, y2 = sample

        self.assertIsNotNone(g1)
        self.assertIsNotNone(g2)
        self.assertEqual(ef1.shape, (len(EXTRA_COLS),))
        self.assertEqual(ef2.shape, (len(EXTRA_COLS),))
        self.assertEqual(gf.shape, (1,))  # GLOBAL_DIM = 1
        self.assertEqual(y1.shape, (NUM_TASKS,))
        self.assertEqual(y2.shape, (NUM_TASKS,))

    def test_dataset_with_none_mol(self):
        """Test dataset with invalid SMILES"""
        df_invalid = self.df.copy()
        # Change first SMILES to invalid value
        # Also need to update mol_1 column since PairDataset will recreate Mol objects from SMILES
        df_invalid.loc[0, 'mol_1'] = Chem.MolFromSmiles('InvalidSMILES')

        dataset = PairDataset(df_invalid, fit_scaler=True, max_dist=self.max_dist)

        # Should filter out invalid samples
        self.assertLess(len(dataset), len(df_invalid))


class TestListDataset(unittest.TestCase):
    """Test ListDataset dataset"""

    def setUp(self):
        # Create list dataset from htl-data-combinations.csv
        df_full = pd.read_csv('htl-data-combinations.csv')
        # Extract first molecule column, using material name as Materials
        self.df = pd.DataFrame({
            'SMILES': df_full['SMILES_1'].head(3),
            'Materials': df_full['mol_1'].head(3).values  # mol_1 column contains material names
        })
        # Add additional feature columns
        extra_cols = [col.replace('_{s}', '') for col in EXTRA_COLS]
        for col in extra_cols:
            self.df[col] = df_full[f'{col}_1'].head(3).values

        self.scaler = StandardScaler()
        self.scaler.fit(np.random.rand(10, len(EXTRA_COLS)))
        self.max_dist = 10

    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = ListDataset(self.df, self.scaler, max_dist=self.max_dist)

        self.assertGreater(len(dataset), 0)
        self.assertEqual(len(dataset), len(self.df))

    def test_dataset_getitem(self):
        """Test dataset sample retrieval"""
        dataset = ListDataset(self.df, self.scaler, max_dist=self.max_dist)

        sample = dataset[0]

        self.assertEqual(len(sample), 4)  # (graph, ef, gf, material)
        graph, ef, gf, material = sample

        self.assertIsNotNone(graph)
        self.assertEqual(ef.shape, (len(EXTRA_COLS),))
        self.assertEqual(gf.shape, (1,))  # GLOBAL_DIM = 1
        self.assertIsInstance(material, str)

    def test_dataset_with_none_mol(self):
        """Test dataset with invalid SMILES"""
        df_invalid = self.df.copy()
        df_invalid.loc[0, 'SMILES'] = 'InvalidSMILES'

        dataset = ListDataset(df_invalid, self.scaler, max_dist=self.max_dist)

        # Should filter out invalid samples
        self.assertLess(len(dataset), len(df_invalid))


class TestCachedPairDataset(unittest.TestCase):
    """Test CachedPairDataset dataset"""

    def setUp(self):
        # Use the first two rows of htl-data-combinations.csv
        df_full = pd.read_csv('htl-data-combinations.csv')
        self.df = df_full.head(2).copy()
        # Convert SMILES to Mol objects
        self.df['mol_1'] = self.df['SMILES_1'].apply(Chem.MolFromSmiles)
        self.df['mol_2'] = self.df['SMILES_2'].apply(Chem.MolFromSmiles)
        self.batch_size = 2
        self.max_dist = 10

    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = CachedPairDataset(self.df, self.batch_size, fit_scaler=True, max_dist=self.max_dist)

        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.scaler)

    def test_dataset_getitem(self):
        """Test dataset sample retrieval"""
        dataset = CachedPairDataset(self.df, self.batch_size, fit_scaler=True, max_dist=self.max_dist)

        batch = dataset[0]

        self.assertEqual(len(batch), 7)  # (mb1, mb2, ef1, ef2, gf, y1, y2)
        mb1, mb2, ef1, ef2, gf, y1, y2 = batch

        self.assertIsNotNone(mb1)
        self.assertIsNotNone(mb2)
        self.assertEqual(ef1.shape[0], self.batch_size)
        self.assertEqual(ef2.shape[0], self.batch_size)
        self.assertEqual(gf.shape[0], self.batch_size)
        self.assertEqual(y1.shape[0], self.batch_size)
        self.assertEqual(y2.shape[0], self.batch_size)


class TestDynamicPairDataset(unittest.TestCase):
    """Test DynamicPairDataset dataset"""

    def setUp(self):
        # Use the first two rows of htl-data-combinations.csv
        df_full = pd.read_csv('htl-data-combinations.csv')
        self.df = df_full.head(2).copy()
        # Convert SMILES to Mol objects
        self.df['mol_1'] = self.df['SMILES_1'].apply(Chem.MolFromSmiles)
        self.df['mol_2'] = self.df['SMILES_2'].apply(Chem.MolFromSmiles)
        self.batch_size = 2
        self.max_dist = 10

    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = DynamicPairDataset(self.df, self.batch_size, fit_scaler=True, max_dist=self.max_dist)

        self.assertGreater(len(dataset), 0)
        self.assertIsNotNone(dataset.scaler)

    def test_dataset_getitem(self):
        """Test dataset sample retrieval"""
        dataset = DynamicPairDataset(self.df, self.batch_size, fit_scaler=True, max_dist=self.max_dist)

        batch = dataset[0]

        self.assertEqual(len(batch), 7)  # (mb1, mb2, ef1, ef2, gf, y1, y2)
        mb1, mb2, ef1, ef2, gf, y1, y2 = batch

        self.assertIsNotNone(mb1)
        self.assertIsNotNone(mb2)
        self.assertEqual(ef1.shape[0], self.batch_size)
        self.assertEqual(ef2.shape[0], self.batch_size)
        self.assertEqual(gf.shape[0], self.batch_size)
        self.assertEqual(y1.shape[0], self.batch_size)
        self.assertEqual(y2.shape[0], self.batch_size)


if __name__ == '__main__':
    unittest.main()
