import { useState, useEffect } from "react";
import SwipeCards, { type CardType } from "./components/SwipeCards";
import entriesData from "./assets/entries.json"; // Adjust the path as needed

// Common fill words to filter out
const fillWords = new Set([
  'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
  'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
  'after', 'above', 'below', 'between', 'among', 'within', 'without',
  'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
  'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
  'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
  'they', 'me', 'him', 'her', 'us', 'them'
]);

const App = () => {
  const [cards, setCards] = useState<CardType[]>([]);
  const [showCards, setShowCards] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchWords, setSearchWords] = useState<Set<string>>(new Set());
  const [showSearchWords, setShowSearchWords] = useState(false);
  const cardsToShow = 7; // Variable to control how many cards to show
  const [nextCardIndex, setNextCardIndex] = useState(cardsToShow); // Start after initial cards
  const [key, setKey] = useState(0); // Force re-render key

  // Add one new card when a card is removed
  useEffect(() => {
    if (showCards && cards.length < cardsToShow) {
      const newCard = entriesData[nextCardIndex % entriesData.length];
      setCards(prevCards => [newCard, ...prevCards]); // Add to bottom of stack
      setNextCardIndex(prev => prev + 1);
    }
  }, [cards.length, showCards, nextCardIndex, cardsToShow]);

  const handleSearch = () => {
    // Only search if there's a query or it's the initial search
    if (searchQuery.trim() || !showCards) {
      // Process search words from the current query
      const currentWords = searchQuery
        .trim()
        .toLowerCase()
        .split(/\s+/)
        .filter(word => word.length > 0 && !fillWords.has(word));
      
      const currentWordsSet = new Set(currentWords);
      setSearchWords(currentWordsSet);
      
      // Get first N entries based on cardsToShow
      const newCards = entriesData.slice(0, cardsToShow).reverse();
      setCards(newCards);
      setShowCards(true);
      setShowSearchWords(true); // Show search words after search is performed
      setKey(prev => prev + 1); // Force SwipeCards to re-render
      setNextCardIndex(cardsToShow); // Reset to start adding from next card
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSearch();
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-neutral-100 px-4">
      {/* Glassy Search Field */}
      <div className="relative w-full max-w-md mb-8 pl-12">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Search for anything..."
          className="w-full h-14 pl-6 pr-14 rounded-full bg-white/70 backdrop-blur-md border border-white/20 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-transparent text-gray-800 placeholder-gray-500 transition-all duration-300"
        />
        {/* Search Icon Button */}
        <button
          onClick={handleSearch}
          className="absolute right-2 top-1/2 transform -translate-y-1/2 h-10 w-10 rounded-full bg-blue-500 hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-400/50 transition-colors duration-200 flex items-center justify-center group"
        >
          <svg 
            className="w-5 h-5 text-white group-hover:scale-110 transition-transform duration-200" 
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" 
            />
          </svg>
        </button>
      </div>

      {/* Search Words Display */}
      {showSearchWords && searchWords.size > 0 && (
        <div className="flex flex-wrap gap-2 mb-4 max-w-md">
          {Array.from(searchWords).map((word, index) => (
            <span
              key={`${word}-${index}`}
              className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-blue-100 text-blue-800 border border-blue-200"
            >
              {word}
            </span>
          ))}
        </div>
      )}

      {showCards && (
        <div className="flex justify-center">
          <SwipeCards key={key} cards={cards} setCards={setCards} />
        </div>
      )}
    </div>
  );
};

export default App;
