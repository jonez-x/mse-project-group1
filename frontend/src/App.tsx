import { useState, useEffect, useRef } from "react";
import { type CardType, type SwipeCardsRef } from "./components/SwipeCards";
import TinderLikeView from "./components/TinderLikeView";
import ClassicListView from "./components/ClassicListView";
import entriesData from "./assets/entries.json"; // Adjust the path as needed
import greenHeartImg from "./assets/img/greenheart.png";
import redx from "./assets/img/redx.png";
import logo from "./assets/img/logo.png";

type ViewMode = 'tinder' | 'list';


const App = () => {
  const [cards, setCards] = useState<CardType[]>([]);
  const [showCards, setShowCards] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [likedCards, setLikedCards] = useState<CardType[]>([]);
  const [dislikedCards, setDislikedCards] = useState<CardType[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('tinder');
  const cardsToShow = 7; // Variable to control how many cards to show
  const [nextCardIndex, setNextCardIndex] = useState(cardsToShow); // Start after initial cards
  const [key, setKey] = useState(0); // Force re-render key
  const swipeCardsRef = useRef<SwipeCardsRef>(null);

  // Add one new card when a card is removed
  useEffect(() => {
    if (showCards && cards.length < cardsToShow) {
      const rawCard = entriesData[nextCardIndex % entriesData.length];
      const newCard: CardType = {
        ...rawCard,
        word_dictionary: rawCard.word_dictionary 
          ? Object.fromEntries(
              Object.entries(rawCard.word_dictionary).filter(([, value]) => value !== undefined)
            ) as Record<string, number>
          : undefined
      };
      setCards(prevCards => [newCard, ...prevCards]); // Add to bottom of stack
      setNextCardIndex(prev => prev + 1);
    }
  }, [cards.length, showCards, nextCardIndex, cardsToShow]);

  const handleSearch = () => {
    // Only search if there's a query or it's the initial search
    if (searchQuery.trim() || !showCards) {
      // Clear liked and disliked arrays when searching
      setLikedCards([]);
      setDislikedCards([]);
      
      // For list view, load all entries; for tinder view, load limited entries
      const entriesToLoad = viewMode === 'list' ? entriesData : entriesData.slice(0, cardsToShow);
      const newCards = entriesToLoad.map(rawCard => ({
        ...rawCard,
        word_dictionary: rawCard.word_dictionary 
          ? Object.fromEntries(
              Object.entries(rawCard.word_dictionary).filter(([, value]) => value !== undefined)
            ) as Record<string, number>
          : undefined
      } as CardType));
      
      // Only reverse for tinder view (to maintain stack order)
      const finalCards = viewMode === 'tinder' ? newCards.reverse() : newCards;
      
      setCards(finalCards);
      setShowCards(true);
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

  const handleDislikeClick = () => {
    swipeCardsRef.current?.swipeLeft();
  };

  const handleLikeClick = () => {
    swipeCardsRef.current?.swipeRight();
  };

  // Add keyboard event listener for arrow keys (only in Tinder mode)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!showCards || viewMode !== 'tinder') return; // Only allow swiping when cards are shown and in Tinder mode
      
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
    if (showCards) {
      // Re-trigger search when switching view modes to load appropriate amount of data
      const entriesToLoad = viewMode === 'list' ? entriesData : entriesData.slice(0, cardsToShow);
      const newCards = entriesToLoad.map(rawCard => ({
        ...rawCard,
        word_dictionary: rawCard.word_dictionary 
          ? Object.fromEntries(
              Object.entries(rawCard.word_dictionary).filter(([, value]) => value !== undefined)
            ) as Record<string, number>
          : undefined
      } as CardType));
      
      // Only reverse for tinder view (to maintain stack order)
      const finalCards = viewMode === 'tinder' ? newCards.reverse() : newCards;
      
      setCards(finalCards);
      setKey(prev => prev + 1); // Force SwipeCards to re-render
    }
  }, [viewMode, showCards, cardsToShow]); // Re-run search when view mode changes

  return (
    
    <div className="min-h-screen flex flex-col bg-neutral-100">
      {/* Header with Logo */}
      <div className="flex justify-between items-center p-6 bg-neutral-100 backdrop-blur-sm border-b border-white/30">
        <div className="w-20"></div> {/* Spacer for centering */}
        <img src={logo} alt="Logo" className="h-10" />
        
        {/* View Toggle Button */}
        <button
          onClick={() => setViewMode(viewMode === 'tinder' ? 'list' : 'tinder')}
          className="w-20 h-10 rounded-full bg-white/90 backdrop-blur-sm border-2 border-gray-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200 text-black"
          
        >
          {viewMode === 'tinder' ? 'Tünder' : 'Classic'}
        </button>
      </div>

      {/* Top Search Bar */}
      <div className="flex justify-center items-center gap-6 p-4 bg-neutral/50 backdrop-blur-sm border-b border-white/20">
        {/* Dislike Button - Only show in Tinder mode */}
        {showCards && viewMode === 'tinder' && (
          <button 
            onClick={handleDislikeClick}
            className="w-14 h-14 rounded-full bg-white/90 backdrop-blur-sm border-2 border-red-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200"
          >
            <img src={redx} alt="dislike button" className="w-8 opacity-70" />
          </button>
        )}
        
        <div className="relative w-full max-w-md">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search for anything..."
            className="w-full h-14 pl-6 pr-14 rounded-full bg-neutral/70 backdrop-blur-md border border-white/20 shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-400/50 focus:border-transparent text-gray-800 placeholder-gray-500 transition-all duration-300"
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

        {/* Like Button - Only show in Tinder mode */}
        {showCards && viewMode === 'tinder' && (
          <button 
            onClick={handleLikeClick}
            className="w-14 h-14 rounded-full bg-white/90 backdrop-blur-sm border-2 border-green-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200"
          >
            <img src={greenHeartImg} alt="like button" className="w-8 opacity-70" />
          </button>
        )}
      </div>

      {/* Main Content Area */}
      {showCards && (
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
            />
          ) : (
            <ClassicListView
              cards={cards}
              searchQuery={searchQuery}
            />
          )}
        </>
      )}
    </div>
  );
};

export default App;
