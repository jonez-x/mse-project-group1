import { useState, useCallback, useRef } from 'react';

export interface AutocompleteSuggestion {
  word: string;
  score: number;
  type: 'completion' | 'next_word';
  model: string;
  full_query: string;
}

export interface UseSimpleAutocompleteReturn {
  suggestions: AutocompleteSuggestion[];
  isLoading: boolean;
  error: string | null;
  getSuggestions: (query: string) => void;
  clearSuggestions: () => void;
}

export const useSimpleAutocomplete = (
  apiBaseUrl: string = 'http://localhost:8000'
): UseSimpleAutocompleteReturn => {
  const [suggestions, setSuggestions] = useState<AutocompleteSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const abortControllerRef = useRef<AbortController | null>(null);

  const getSuggestions = useCallback(async (query: string) => {
    if (!query.trim()) {
      setSuggestions([]);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Cancel previous request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      abortControllerRef.current = new AbortController();

      const params = new URLSearchParams({
        q: query,
        max_suggestions: '5'
      });

      const response = await fetch(
        `${apiBaseUrl}/autocomplete?${params}`,
        {
          signal: abortControllerRef.current.signal,
          headers: {
            'Accept': 'application/json',
          }
        }
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setSuggestions(data.suggestions || []);
      
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        return; // Request was canceled
      }
      
      console.error('Autocomplete error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      setSuggestions([]);
    } finally {
      setIsLoading(false);
    }
  }, [apiBaseUrl]);

  const clearSuggestions = useCallback(() => {
    setSuggestions([]);
    setError(null);
    setIsLoading(false);
    
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  }, []);

  return {
    suggestions,
    isLoading,
    error,
    getSuggestions,
    clearSuggestions
  };
};