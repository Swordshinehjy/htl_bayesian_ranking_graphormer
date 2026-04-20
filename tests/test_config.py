
"""
Configuration class unit tests
"""
import unittest
from dataclasses import asdict
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from htl_package import ModelConfig, TrainingConfig, FinetuneConfig, PredictConfig


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig configuration"""

    def test_default_values(self):
        """Test default values are set correctly"""
        config = ModelConfig()
        self.assertEqual(config.hidden_size, 300)
        self.assertEqual(config.depth, 3)
        self.assertEqual(config.num_heads, 6)
        self.assertEqual(config.dropout, 0.0)
        self.assertEqual(config.ffn_hidden, 256)
        self.assertEqual(config.max_degree, 15)
        self.assertEqual(config.max_dist, 15)
        self.assertTrue(config.auto_compute_stats)

    def test_custom_values(self):
        """Test custom values are set correctly"""
        config = ModelConfig(
            hidden_size=256,
            depth=4,
            num_heads=8,
            dropout=0.1,
            ffn_hidden=512
        )
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.depth, 4)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.ffn_hidden, 512)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = ModelConfig(hidden_size=256, depth=4)
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['hidden_size'], 256)
        self.assertEqual(config_dict['depth'], 4)

    def test_from_dict(self):
        """Test creation from dictionary"""
        config_dict = {
            'hidden_size': 256,
            'depth': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'ffn_hidden': 512
        }
        config = ModelConfig.from_dict(config_dict)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.depth, 4)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.ffn_hidden, 512)

    def test_from_dict_excludes_invalid_fields(self):
        """Test creation from dictionary excludes invalid fields"""
        config_dict = {
            'hidden_size': 256,
            'depth': 4,
            'invalid_field': 'should_be_ignored'
        }
        config = ModelConfig.from_dict(config_dict)
        self.assertEqual(config.hidden_size, 256)
        self.assertEqual(config.depth, 4)
        self.assertFalse(hasattr(config, 'invalid_field'))


class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig configuration"""

    def test_default_values(self):
        """Test default values are set correctly"""
        config = TrainingConfig()
        self.assertEqual(config.csv_path, "htl-data-combinations.csv")
        self.assertEqual(config.save_dir, "checkpoints")
        self.assertEqual(config.epochs, 1000)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.lr, 5e-4)
        self.assertEqual(config.weight_decay, 1e-5)
        self.assertEqual(config.patience, 50)
        self.assertEqual(config.early_stop_warmup, 20)
        self.assertEqual(config.val_ratio, 0.1)
        self.assertEqual(config.test_ratio, 0.1)
        self.assertEqual(config.split, "random")
        self.assertIsNone(config.n_cv_folds)
        self.assertEqual(config.seed, 42)
        self.assertTrue(config.cache_val_test)

    def test_custom_values(self):
        """Test custom values are set correctly"""
        config = TrainingConfig(
            epochs=500,
            batch_size=16,
            lr=1e-3,
            patience=30,
            early_stop_warmup=10,
            split="group",
            n_cv_folds=5,
        )
        self.assertEqual(config.epochs, 500)
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.lr, 1e-3)
        self.assertEqual(config.patience, 30)
        self.assertEqual(config.early_stop_warmup, 10)
        self.assertEqual(config.split, "group")
        self.assertEqual(config.n_cv_folds, 5)


class TestFinetuneConfig(unittest.TestCase):
    """Test FinetuneConfig configuration"""

    def test_default_values(self):
        """Test default values are set correctly"""
        config = FinetuneConfig()
        self.assertEqual(config.csv_path, "htl-data-combinations.csv")
        self.assertEqual(config.checkpoint_path, "checkpoints/best_model.pt")
        self.assertEqual(config.save_dir, "checkpoints")
        self.assertEqual(config.finetune_epochs, 10)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.lr, 1e-5)
        self.assertEqual(config.weight_decay, 1e-6)
        self.assertEqual(config.seed, 42)


class TestPredictConfig(unittest.TestCase):
    """Test PredictConfig configuration"""

    def test_default_values(self):
        """Test default values are set correctly"""
        config = PredictConfig()
        self.assertEqual(config.predict_csv, "")
        self.assertEqual(config.checkpoint_path, "checkpoints/best_model.pt")
        self.assertEqual(config.output_path, "predictions.csv")


if __name__ == '__main__':
    unittest.main()
