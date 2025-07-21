import unittest
from unittest.mock import patch, MagicMock
from autocomplete_system.services.autocomplete import AutocompleteService, ModelType
from autocomplete_system.models.base import AutocompleteResult


class TestAutocompleteService(unittest.TestCase):
    """Test cases for the AutocompleteService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the model initialization to avoid loading actual models
        with patch.object(AutocompleteService, '_initialize_models'):
            self.service = AutocompleteService(default_model=ModelType.DATAMUSE)

        # Set up mock models
        self.mock_datamuse = MagicMock()
        self.mock_ngram = MagicMock()

        self.service.models = {
            ModelType.DATAMUSE: self.mock_datamuse,
            ModelType.NGRAM: self.mock_ngram
        }

    def test_initialization(self):
        """Test service initialization."""
        self.assertEqual(self.service.default_model, ModelType.DATAMUSE)
        self.assertIn(ModelType.DATAMUSE, self.service.models)
        self.assertIn(ModelType.NGRAM, self.service.models)

    def test_get_suggestions_with_default_model(self):
        """Test getting suggestions with default model."""
        # Mock response from datamuse model
        mock_results = [
            AutocompleteResult(
                word="test",
                score=1.0,
                type="completion",
            ),
            AutocompleteResult(
                word="testing",
                score=0.8,
                type="completion",
            )
        ]
        self.mock_datamuse.suggest.return_value = mock_results

        suggestions = self.service.get_suggestions(
            query="tes",
            max_suggestions=5,
        )

        self.assertEqual(len(suggestions), 2)
        self.assertEqual(suggestions[0]["word"], "test")
        self.assertEqual(suggestions[0]["model"], "datamuse")
        self.assertEqual(suggestions[0]["type"], "completion")
        self.assertIn("full_query", suggestions[0])

    def test_get_suggestions_with_specific_model(self):
        """Test getting suggestions with specific model."""
        mock_results = [AutocompleteResult(word="example", score=1.0, type="next_word")]
        self.mock_ngram.suggest.return_value = mock_results

        suggestions = self.service.get_suggestions(
            query="hello ",
            model_type=ModelType.NGRAM,
            max_suggestions=3,
        )

        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["word"], "example")
        self.assertEqual(suggestions[0]["model"], "ngram")
        self.assertEqual(suggestions[0]["type"], "next_word")

    def test_get_suggestions_empty_query(self):
        """Test getting suggestions with empty query."""
        suggestions = self.service.get_suggestions(
            query="",
            max_suggestions=5,
        )
        self.assertEqual(suggestions, [])

    def test_get_suggestions_unavailable_model(self):
        """Test getting suggestions with unavailable model."""
        self.service.models[ModelType.NGRAM] = None

        suggestions = self.service.get_suggestions(
            "test",
            model_type=ModelType.NGRAM,
            max_suggestions=3,
        )

        self.assertEqual(suggestions, [])

    def test_get_available_models(self):
        """Test getting list of available models."""
        available = self.service.get_available_models()

        self.assertIn("datamuse", available)
        self.assertIn("ngram", available)

    def test_get_available_models_with_none_model(self):
        """Test getting available models when some are None."""
        self.service.models[ModelType.NGRAM] = None

        available = self.service.get_available_models()

        self.assertIn("datamuse", available)
        self.assertNotIn("ngram", available)

    def test_get_model_info(self):
        """Test getting model information."""
        mock_info = {"name": "test_model", "vocab_size": 1000}
        self.mock_datamuse.get_model_info.return_value = mock_info

        info = self.service.get_model_info(model_type=ModelType.DATAMUSE)

        self.assertEqual(info, mock_info)
        self.mock_datamuse.get_model_info.assert_called_once()

    def test_get_model_info_unavailable_model(self):
        """Test getting info for unavailable model."""
        self.service.models[ModelType.NGRAM] = None

        info = self.service.get_model_info(ModelType.NGRAM)

        self.assertIsNone(info)

    def test_full_query_generation_completion(self):
        """Test full query generation for word completion."""
        mock_results = [AutocompleteResult(
            word="testing",
            score=1.0,
            type="completion"),
        ]
        self.mock_datamuse.suggest.return_value = mock_results

        suggestions = self.service.get_suggestions(
            query="hello tes",
            max_suggestions=1,
        )

        self.assertEqual(suggestions[0]["full_query"], "hello testing")

    def test_full_query_generation_next_word(self):
        """Test full query generation for next word prediction."""
        mock_results = [AutocompleteResult(
            word="world",
            score=1.0,
            type="next_word"),
        ]
        self.mock_datamuse.suggest.return_value = mock_results

        suggestions = self.service.get_suggestions(
            query="hello ",
            max_suggestions=1,
        )

        self.assertEqual(suggestions[0]["full_query"], "hello world")


if __name__ == '__main__':
    unittest.main()
