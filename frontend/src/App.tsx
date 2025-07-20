import { useState, useEffect, useRef } from "react";
import { type CardType, type SwipeCardsRef } from "./components/SwipeCards";
import TinderLikeView from "./components/TinderLikeView";
import ClassicListView from "./components/ClassicListView";
import greenHeartImg from "./assets/img/greenheart.png";
import redx from "./assets/img/redx.png";
import logo from "./assets/img/logo.png";

type ViewMode = 'tinder' | 'list';

// API response types
interface ApiDoc {
  id: number;
  url: string;
  favicon?: string;
  title?: string;
  description?: string;
  image?: string;
  word_dictionary?: Record<string, number>;
}

interface SearchResponse {
  results: ApiDoc[];
}

const API_BASE_URL = 'http://localhost:8000';

const App = () => {
  const [cards, setCards] = useState<CardType[]>([]);
  const [allSearchResults, setAllSearchResults] = useState<CardType[]>([]);
  const [showCards, setShowCards] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [likedCards, setLikedCards] = useState<CardType[]>([]);
  const [dislikedCards, setDislikedCards] = useState<CardType[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('tinder');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoOpenPages, setAutoOpenPages] = useState(false);
  const [isSearchEngineReady, setIsSearchEngineReady] = useState(false);
  const cardsToShow = 7; // Variable to control how many cards to show
  const [key, setKey] = useState(0); // Force re-render key
  const swipeCardsRef = useRef<SwipeCardsRef>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Check if search engine is ready on component mount
  useEffect(() => {
    const checkSearchEngineStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/v2/search?q=test`);
        if (response.ok) {
          setIsSearchEngineReady(true);
        }
      } catch (error) {
        console.log('Search engine not ready yet, will retry...');
        // Retry every 2 seconds until ready
        setTimeout(checkSearchEngineStatus, 2000);
      }
    };
    
    checkSearchEngineStatus();
  }, []);

  // API function to search
  const searchAPI = async (query: string, version: string = 'v2'): Promise<CardType[]> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/${version}/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      const data: SearchResponse = await response.json();
      
      // Convert API response to CardType format
      const cards: CardType[] = data.results.map(doc => ({
        id: doc.id,
        url: doc.url,
        favicon: doc.favicon || `https://www.google.com/s2/favicons?domain=${new URL(doc.url).hostname}&sz=16`,
        title: doc.title,
        description: doc.description,
        image: doc.image,
        word_dictionary: doc.word_dictionary
      }));
      
      return cards;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      return [];
    } finally {
      setIsLoading(false);
    }
  };

  // Add one new card when a card is removed (for Tinder mode)
  useEffect(() => {
    if (showCards && viewMode === 'tinder' && cards.length < cardsToShow && allSearchResults.length > 0) {
      const availableCards = allSearchResults.filter(result => 
        !cards.some(card => card.id === result.id) &&
        !likedCards.some(card => card.id === result.id) &&
        !dislikedCards.some(card => card.id === result.id)
      );
      
      if (availableCards.length > 0) {
        const nextCard = availableCards[0];
        setCards(prevCards => [nextCard, ...prevCards]);
      }
    }
  }, [cards, showCards, viewMode, allSearchResults, likedCards, dislikedCards, cardsToShow]);

  const launchSearch = async () => {
    // Only search if there's a query and search engine is ready
    if (!searchQuery.trim() || !isSearchEngineReady) return;

    // Remove focus from input to hide caret
    inputRef.current?.blur();
    
    // Clear liked and disliked arrays when searching
    setLikedCards([]);
    setDislikedCards([]);
    
    try {
      const searchResults = await searchAPI(searchQuery);
      
      if (searchResults.length > 0) {
        setAllSearchResults(searchResults);
        
        if (viewMode === 'tinder') {
          // For Tinder mode, show first few cards
          const initialCards = searchResults.slice(0, cardsToShow).reverse();
          setCards(initialCards);
        } else {
          // For list mode, show all results
          setCards(searchResults);
        }
        
        setShowCards(true);
        setKey(prev => prev + 1); // Force SwipeCards to re-render
      }
    } catch (err) {
      console.error('Search failed:', err);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      launchSearch();
    }
  };

  const handleSearchClick = () => launchSearch();

    const handleDislikeClick = () => {
        swipeCardsRef.current?.swipeLeft();
    };

    const handleLikeClick = () => {
        swipeCardsRef.current?.swipeRight();
    };

    // Add keyboard event listener for arrow keys (only in Tinder mode)
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            // Don't allow swiping if user is typing in the search input
            const activeElement = document.activeElement;
            const isTypingInInput = activeElement && activeElement.tagName === 'INPUT';
            
            if (!showCards || viewMode !== 'tinder' || isTypingInInput) return; // Only allow swiping when cards are shown and in Tinder mode and not typing

            if (e.key === "ArrowLeft") {
                e.preventDefault();
                handleDislikeClick();
            } else if (e.key === "ArrowRight") {
                e.preventDefault();
                handleLikeClick();
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        // Cleanup event listener on component unmount
        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
  }, [showCards, viewMode]); // Re-run effect when showCards or viewMode changes

  // Reload data when view mode changes
  useEffect(() => {
    if (showCards && allSearchResults.length > 0) {
      if (viewMode === 'tinder') {
        // For Tinder mode, show limited cards
        const initialCards = allSearchResults.slice(0, cardsToShow).reverse();
        setCards(initialCards);
      } else {
        // For list mode, show all search results
        setCards(allSearchResults);
      }
      setKey(prev => prev + 1); // Force SwipeCards to re-render
    }
  }, [viewMode, showCards, allSearchResults, cardsToShow]); // Re-run when view mode changes

    return (

        <div className="min-h-screen flex flex-col bg-neutral-100">
            {/* Header with Logo */}
            <div
                className="flex justify-between items-center p-6 bg-neutral-100 backdrop-blur-sm border-b border-white/30">
                {/* Auto-open Pages Toggle - Only show in Tinder mode */}
                {viewMode === 'tinder' ? (
                    <div
                        className="flex items-center gap-2 bg-white/90 backdrop-blur-sm border-2 border-gray-200/70 shadow-lg rounded-full px-3 py-2">
                        <span className="text-xs text-gray-700 whitespace-nowrap">Auto-open:</span>
                        <button
                            onClick={() => setAutoOpenPages(!autoOpenPages)}
                            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                                autoOpenPages ? 'bg-blue-600' : 'bg-gray-300'
                            }`}
                        >
              <span
                  className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                      autoOpenPages ? 'translate-x-5' : 'translate-x-1'
                  }`}
              />
                        </button>
                    </div>
                ) : (
                    <div className="w-20"></div>
                )}
                {/* Spacer for centering when not in Tinder mode */}
                <img src={logo} alt="Logo" className="h-10"/>

                {/* View Toggle Button */}
                <button
                    onClick={() => setViewMode(viewMode === 'tinder' ? 'list' : 'tinder')}
                    className="w-20 h-10 rounded-full bg-white/90 backdrop-blur-sm border-2 border-gray-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200 text-black"

                >
                    {viewMode === 'tinder' ? 'Tünder' : 'Classic'}
                </button>
            </div>

            {/* Top Search Bar */}
            <div
                className="flex justify-center items-center gap-6 p-4 bg-neutral/50 backdrop-blur-sm border-b border-white/20">
                {/* Dislike Button - Only show in Tinder mode */}
                {showCards && viewMode === 'tinder' && (
                    <button
                        onClick={handleDislikeClick}
                        className="w-14 h-14 rounded-full bg-white/90 backdrop-blur-sm border-2 border-red-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200"
                    >
                        <img src={redx} alt="dislike button" className="w-8 opacity-70"/>
                    </button>
                )}

                <div className="relative w-full max-w-md">
                    <input
                        ref={inputRef}
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder={isSearchEngineReady ? "Search for anything..." : "Search engine is initializing..."}
                        disabled={!isSearchEngineReady || isLoading}
                        className={`w-full h-14 pl-6 pr-14 rounded-full backdrop-blur-md border shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-transparent text-gray-800 placeholder-gray-500 transition-all duration-300 ${
                            !isSearchEngineReady || isLoading 
                                ? 'bg-gray-300/70 border-gray-200/20 cursor-not-allowed' 
                                : 'bg-neutral/70 border-white/20'
                        }`}
                    />
                    {/* Search Icon Button */}
                    <button
                        onClick={handleSearchClick}
                        disabled={!isSearchEngineReady || isLoading}
                        className={`absolute right-2 top-1/2 transform -translate-y-1/2 h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-400/50 transition-colors duration-200 flex items-center justify-center group ${
                            !isSearchEngineReady || isLoading
                                ? 'bg-gray-400 cursor-not-allowed'
                                : 'bg-blue-500 hover:bg-blue-600'
                        }`}
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

                {/* Like Button - Only show in Tinder mode */}
                {showCards && viewMode === 'tinder' && (
                    <button
                        onClick={handleLikeClick}
                        className="w-14 h-14 rounded-full bg-white/90 backdrop-blur-sm border-2 border-green-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200"
                    >
                        <img src={greenHeartImg} alt="like button" className="w-8 opacity-70"/>
                    </button>
                )}
            </div>

      {/* Main Content Area */}
      {!isSearchEngineReady && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <h2 className="text-xl font-semibold text-gray-700 mb-2">Initializing Search Engine</h2>
            <p className="text-gray-600 max-w-md">
              Loading and indexing documents, training models... This may take a moment.
            </p>
            <div className="text-sm text-gray-400 mt-4">
              The search engine is warming up. You'll be able to search once it's ready!
            </div>
          </div>
        </div>
      )}

      {isSearchEngineReady && isLoading && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Searching...</p>
          </div>
        </div>
      )}
      
      {isSearchEngineReady && error && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-red-500 text-6xl mb-4">⚠️</div>
            <h3 className="text-xl font-medium text-gray-900 mb-2">Search Error</h3>
            <p className="text-red-600 mb-4">{error}</p>
            <button
              onClick={() => setError(null)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      )}

      {isSearchEngineReady && showCards && !isLoading && !error && (
        <>
          {viewMode === 'tinder' ? (
            <TinderLikeView
              cards={cards}
              setCards={setCards}
              likedCards={likedCards}
              dislikedCards={dislikedCards}
              onLike={(card) => setLikedCards(prev => [card, ...prev])}
              onDislike={(card) => setDislikedCards(prev => [card, ...prev])}
              swipeCardsRef={swipeCardsRef}
              componentKey={key}
              autoOpenPages={autoOpenPages}
            />
          ) : (
            <ClassicListView
              cards={cards}
              searchQuery={searchQuery}
            />
          )}
        </>
      )}
      
      {isSearchEngineReady && !showCards && !isLoading && !error && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="text-gray-400 text-6xl mb-4">🔍</div>
            <h3 className="text-xl font-medium text-gray-900 mb-2">Ready to Search</h3>
            <p className="text-gray-600">Enter a search query above to get started</p>
          </div>
        </div>
      )}
        </div>
    );
};

export default App;
