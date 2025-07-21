import unittest
from autocomplete_system.models.ngram import NgramModel
from autocomplete_system.models.base import AutocompleteResult


class TestNgramModel(unittest.TestCase):
    """Test cases for the NgramModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = NgramModel(n=3)
        self.test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown fox is very fast and agile",
            "The lazy dog sleeps under the tree",
            "Fast cars drive on the highway"
        ]

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.n, 3)
        self.assertEqual(self.model.name, "ngram_3")
        self.assertFalse(self.model.is_trained)
        self.assertEqual(self.model.vocab_size, 0)

    def test_tokenization(self):
        """Test text tokenization."""
        text = "Hello, world! This is a test."
        tokens = self.model._tokenize(text)
        expected = ["hello", "world", "this", "is", "test"]
        self.assertEqual(tokens, expected)

    def test_training(self):
        """Test model training."""
        self.model.train(
            texts=self.test_texts,
            min_freq=1,
            verbose=False,
        )

        self.assertTrue(self.model.is_trained)
        self.assertGreater(self.model.vocab_size, 0)
        self.assertGreater(self.model.training_time, 0)

        # Check that n-grams were built
        for k in range(1, self.model.n + 1):
            self.assertGreater(len(self.model.ngrams[k]), 0)

    def test_word_completion(self):
        """Test word completion functionality."""
        self.model.train(
            texts=self.test_texts,
            min_freq=1,
            verbose=False,
        )

        # Test completion for "qui" -> should suggest "quick"
        results = self.model._complete_word(
            partial_word="qui",
            n_suggestions=3,
        )
        self.assertIsInstance(results, list)

        if results:  # If suggestions found
            self.assertIsInstance(results[0], AutocompleteResult)
            self.assertEqual(results[0].type, "completion")
            self.assertTrue("quick" in [r.word for r in results])

    def test_next_word_prediction(self):
        """Test next word prediction."""
        self.model.train(
            texts=self.test_texts,
            min_freq=1,
            verbose=False,
        )

        # Test next word after "the"
        results = self.model._predict_next_word(
            words=["the"],
            n_suggestions=3,
        )
        self.assertIsInstance(results, list)

        if results:  # If suggestions found
            self.assertIsInstance(results[0], AutocompleteResult)
            self.assertEqual(results[0].type, "next_word")

    def test_suggest_method(self):
        """Test the main suggest method."""
        self.model.train(
            texts=self.test_texts,
            min_freq=1,
            verbose=False,
        )

        # Test word completion
        completion_results = self.model.suggest(
            query="qui",
            n_suggestions=3,
        )
        self.assertIsInstance(completion_results, list)

        # Test next word prediction
        next_word_results = self.model.suggest(
            query="the ",
            n_suggestions=3,
        )
        self.assertIsInstance(next_word_results, list)

        # Test empty query
        empty_results = self.model.suggest(
            query="",
            n_suggestions=3,
        )
        self.assertEqual(empty_results, [])

    def test_untrained_model_suggestions(self):
        """Test that untrained model returns empty suggestions."""
        results = self.model.suggest(
            query="test",
            n_suggestions=3,
        )
        self.assertEqual(results, [])

    def test_get_model_info(self):
        """Test model info retrieval."""
        self.model.train(
            texts=self.test_texts,
            min_freq=1,
            verbose=False,
        )

        info = self.model.get_model_info()
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("vocab_size", info)
        self.assertIn("training_time", info)
        self.assertIn("is_trained", info)
        self.assertIn("n", info)

        self.assertEqual(info["name"], "ngram_3")
        self.assertTrue(info["is_trained"])
        self.assertEqual(info["n"], 3)

    def test_is_training_required(self):
        """Test training requirement check."""
        self.assertTrue(self.model.is_training_required())


if __name__ == '__main__':
    unittest.main()
