import React, { useState, useRef, useEffect } from 'react';
import { useSimpleAutocomplete, type AutocompleteSuggestion } from '../hooks/useSimpleAutocomplete';

interface SimpleAutocompleteProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: (value: string) => void;
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export const SimpleAutocomplete: React.FC<SimpleAutocompleteProps> = ({
  value,
  onChange,
  onSubmit,
  placeholder = "Search for anything...",
  className = "",
  disabled = false
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  
  const inputRef = useRef<HTMLInputElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const { suggestions, isLoading, error, getSuggestions, clearSuggestions } = useSimpleAutocomplete();

  // Debounced autocomplete
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    onChange(newValue);
    
    // Clear previous timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    
    if (newValue.trim().length >= 2) {
      // Debounce the API call
      timeoutRef.current = setTimeout(() => {
        getSuggestions(newValue);
        setShowSuggestions(true);
        setSelectedIndex(-1);
      }, 300);
    } else {
      clearSuggestions();
      setShowSuggestions(false);
    }
  };

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!showSuggestions || suggestions.length === 0) {
      if (e.key === 'Enter') {
        e.preventDefault();
        onSubmit?.(value);
      }
      return;
    }

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev < suggestions.length - 1 ? prev + 1 : 0
        );
        break;
        
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => 
          prev > 0 ? prev - 1 : suggestions.length - 1
        );
        break;
        
      case 'Enter':
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
          selectSuggestion(suggestions[selectedIndex]);
        } else {
          // Hide suggestions when submitting
          setShowSuggestions(false);
          setSelectedIndex(-1);
          onSubmit?.(value);
        }
        break;
        
      case 'Escape':
        e.preventDefault();
        setShowSuggestions(false);
        setSelectedIndex(-1);
        break;
        
      case ' ': // Space bar
      case 'ArrowRight': // Right arrow key
        if (selectedIndex >= 0 && selectedIndex < suggestions.length) {
          e.preventDefault();
          selectSuggestion(suggestions[selectedIndex]);
        }
        break;
    }
  };

  // Select a suggestion
  const selectSuggestion = (suggestion: AutocompleteSuggestion) => {
    onChange(suggestion.full_query);
    setShowSuggestions(false);
    setSelectedIndex(-1);
    inputRef.current?.focus();
  };

  // Close suggestions when clicking outside
  const handleBlur = () => {
    setTimeout(() => {
      setShowSuggestions(false);
      setSelectedIndex(-1);
    }, 150);
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div className="relative w-full">
      <input
        ref={inputRef}
        type="text"
        value={value}
        onChange={handleInputChange}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        placeholder={placeholder}
        disabled={disabled}
        className={`w-full h-14 pl-6 pr-14 rounded-full bg-neutral/70 backdrop-blur-md border border-white/20 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-transparent text-gray-800 placeholder-gray-500 transition-all duration-300 ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        } ${className}`}
      />
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute right-16 top-1/2 transform -translate-y-1/2">
          <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
        </div>
      )}


      {/* Suggestions dropdown */}
      {showSuggestions && suggestions.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-xl border border-gray-200 z-[9999] max-h-64 overflow-y-auto">
          
          {suggestions.map((suggestion, index) => (
            <div
              key={`${suggestion.full_query}-${index}`}
              className={`px-4 py-3 cursor-pointer transition-colors duration-150 ${
                index === selectedIndex 
                  ? 'bg-blue-50 text-blue-700 border-l-4 border-blue-500' 
                  : 'hover:bg-gray-50 text-gray-700'
              } ${index === 0 ? 'rounded-t-lg' : ''} ${index === suggestions.length - 1 ? 'rounded-b-lg' : 'border-b border-gray-100'}`}
              onMouseDown={(e) => {
                e.preventDefault();
                selectSuggestion(suggestion);
              }}
              onMouseEnter={() => setSelectedIndex(index)}
            >
              <span className="font-medium text-gray-800">{suggestion.full_query}</span>
            </div>
          ))}
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="absolute top-full left-0 right-0 mt-2 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm z-[9999]">
          <span className="font-medium">Error:</span> {error}
        </div>
      )}
    </div>
  );
};