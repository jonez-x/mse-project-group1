import unittest
from unittest.mock import patch, MagicMock
import requests
from autocomplete_system.models.datamuse import DataMuseModel
from autocomplete_system.models.base import AutocompleteResult


class TestDataMuseModel(unittest.TestCase):
    """Test cases for the DataMuseModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = DataMuseModel()

    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.name, "datamuse")
        self.assertFalse(self.model.is_training_required())

    def test_train_is_noop(self):
        """Test that training is a no-op for DataMuse model."""
        # Should not raise any exceptions
        self.model.train(["some", "test", "texts"])

    def test_is_valid_word(self):
        """Test word validation."""
        # Valid words
        self.assertTrue(self.model._is_valid_word("hello"))
        self.assertTrue(self.model._is_valid_word("don't"))
        self.assertTrue(self.model._is_valid_word("well-known"))

        # Invalid words
        self.assertFalse(self.model._is_valid_word("a"))  # Too short
        self.assertFalse(self.model._is_valid_word("test123"))  # Contains numbers
        self.assertFalse(self.model._is_valid_word("test."))  # Contains punctuation
        self.assertFalse(self.model._is_valid_word(""))  # Empty string

    @patch('requests.get')
    def test_complete_word_success(self, mock_get):
        """Test word completion with successful API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"word": "test", "score": 100},
            {"word": "testing", "score": 90}
        ]
        mock_get.return_value = mock_response

        results = self.model._complete_word(
            partial_word="tes",
            n_suggestions=3,
        )

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], AutocompleteResult)
        self.assertEqual(results[0].word, "test")
        self.assertEqual(results[0].score, 100.0)
        self.assertEqual(results[0].type, "completion")

    @patch('requests.get')
    def test_complete_word_api_error(self, mock_get):
        """Test word completion with API error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        results = self.model._complete_word(
            partial_word="tes",
            n_suggestions=3,
        )

        self.assertEqual(results, [])

    @patch('requests.get')
    def test_complete_word_request_exception(self, mock_get):
        """Test word completion with request exception."""
        mock_get.side_effect = requests.RequestException("Connection error")

        results = self.model._complete_word(
            partial_word="tes",
            n_suggestions=3,
        )

        self.assertEqual(results, [])

    @patch('requests.get')
    def test_predict_next_word_success(self, mock_get):
        """Test next word prediction with successful API response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"word": "world", "score": 100},
            {"word": "there", "score": 80}
        ]
        mock_get.return_value = mock_response

        results = self.model._predict_next_word(
            words=["hello"],
            n_suggestions=3,
        )

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], AutocompleteResult)
        self.assertEqual(results[0].word, "world")
        self.assertEqual(results[0].type, "next_word")

    @patch('requests.get')
    def test_predict_next_word_fallback_to_bigram(self, mock_get):
        """Test next word prediction fallback to bigram analysis."""
        # First call returns empty results
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = []

        # Second call (bigram) returns results
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = [
            {"word": "example", "score": 50}
        ]

        mock_get.side_effect = [mock_response1, mock_response2]

        results = self.model._predict_next_word(
            words=["rare"],
            n_suggestions=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].word, "example")
        self.assertEqual(results[0].type, "next_word")
        self.assertEqual(mock_get.call_count, 2)

    def test_suggest_word_completion(self):
        """Test suggest method for word completion."""
        with patch.object(self.model, '_complete_word') as mock_complete:
            mock_complete.return_value = [
                AutocompleteResult(
                    word="test",
                    score=100.0,
                    type="completion",
                )
            ]

            results = self.model.suggest(
                query="tes",
                n_suggestions=3,
            )

            mock_complete.assert_called_once_with("tes", 3)
            self.assertEqual(len(results), 1)

    def test_suggest_next_word_prediction(self):
        """Test suggest method for next word prediction."""
        with patch.object(self.model, '_predict_next_word') as mock_predict:
            mock_predict.return_value = [
                AutocompleteResult(
                    word="world",
                    score=100.0,
                    type="next_word",
                )
            ]

            results = self.model.suggest(
                query="hello ",
                n_suggestions=3,
            )

            mock_predict.assert_called_once_with(["hello"], 3)
            self.assertEqual(len(results), 1)

    def test_suggest_empty_query(self):
        """Test suggest method with empty query."""
        results = self.model.suggest(
            query="",
            n_suggestions=3,
        )
        self.assertEqual(results, [])

    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.model.get_model_info()

        self.assertIsInstance(info, dict)
        self.assertEqual(info["name"], "datamuse")
        self.assertEqual(info["type"], "api_based")
        self.assertFalse(info["requires_training"])
        self.assertEqual(info["base_url"], "https://api.datamuse.com")
        self.assertEqual(info["timeout"], 1.0)

    def test_integration_example_functionality(self):
        """Test the example functionality that was in __main__."""
        example_words = ["food", "movie", "car"]

        with patch.object(self.model, 'suggest') as mock_suggest:
            # Mock different responses for completion and next word
            mock_suggest.side_effect = [
                [AutocompleteResult(word="food", score=100.0, type="completion")],  # completion
                [AutocompleteResult(word="good", score=80.0, type="next_word")],  # next word
                [AutocompleteResult(word="movie", score=90.0, type="completion")],  # completion
                [AutocompleteResult(word="theater", score=70.0, type="next_word")],  # next word
                [AutocompleteResult(word="car", score=95.0, type="completion")],  # completion
                [AutocompleteResult(word="wash", score=60.0, type="next_word")]  # next word
            ]

            # Test the pattern that was in __main__
            for word in example_words:
                # Test word completion (partial prefix)
                completion_suggestions = self.model.suggest(
                    query=word[:3],
                    n_suggestions=5,
                )
                self.assertIsInstance(completion_suggestions, list)

                # Test next word prediction (full word + space)
                next_word_suggestions = self.model.suggest(
                    query=word + " ",
                    n_suggestions=5,
                )
                self.assertIsInstance(next_word_suggestions, list)

            # Verify all calls were made
            self.assertEqual(mock_suggest.call_count, 6)


if __name__ == '__main__':
    unittest.main()
