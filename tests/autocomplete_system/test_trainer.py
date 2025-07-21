import unittest
import tempfile
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

from autocomplete_system.trainer import train_and_serialize_model, load_model
from autocomplete_system.models.ngram import NgramModel


class TestTrainer(unittest.TestCase):
    """Test cases for the trainer module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_texts = [
            "This is a simple test document for training",
            "Another document with some words for testing",
            "Test data should help train the model properly"
        ]
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for file in self.temp_dir.glob("*.pkl"):
            file.unlink()
        self.temp_dir.rmdir()

    def test_train_and_serialize_model(self):
        """Test model training and serialization."""
        with patch('autocomplete_system.trainer.SERIALIZED_DIR', self.temp_dir):
            # Train model with test data
            model = train_and_serialize_model(
                model_name="ngram",
                texts=self.test_texts,
                verbose=False,
            )

            # Check model was trained
            self.assertIsInstance(model, NgramModel)
            self.assertTrue(model.is_trained)
            self.assertGreater(model.vocab_size, 0)

            # Check file was created
            model_file = self.temp_dir / "ngram.pkl"
            self.assertTrue(model_file.exists())

    def test_load_model(self):
        """Test model loading from disk."""
        # Create and save a test model
        test_model = NgramModel(n=2)
        test_model.train(
            texts=self.test_texts,
            verbose=False,
        )

        model_file = self.temp_dir / "test_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(test_model, f)

        # Mock the SERIALIZED_DIR to point to our temp directory
        with patch('autocomplete_system.trainer.SERIALIZED_DIR', self.temp_dir):
            loaded_model = load_model(
                model_name="test_model",
                verbose=False,
            )

        # Verify loaded model
        self.assertIsInstance(loaded_model, NgramModel)
        self.assertTrue(loaded_model.is_trained)
        self.assertEqual(loaded_model.vocab_size, test_model.vocab_size)

    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        with patch('autocomplete_system.trainer.SERIALIZED_DIR', self.temp_dir):
            with self.assertRaises(FileNotFoundError):
                load_model(
                    model_name="nonexistent_model",
                    verbose=False,
                )


if __name__ == '__main__':
    unittest.main()
