import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import SwipeCards, { type CardType, type SwipeCardsRef } from "./components/SwipeCards";
import entriesData from "./assets/entries.json"; // Adjust the path as needed
import greenHeartImg from "./assets/img/greenheart.png";
import redx from "./assets/img/redx.png";
import leftarrow from "./assets/img/leftarrow.png";
import rightarrow from "./assets/img/rightarrow.png";


const App = () => {
  const [cards, setCards] = useState<CardType[]>([]);
  const [showCards, setShowCards] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [likedCards, setLikedCards] = useState<CardType[]>([]);
  const [dislikedCards, setDislikedCards] = useState<CardType[]>([]);
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
      // Get first N entries based on cardsToShow and clean the data
      const rawCards = entriesData.slice(0, cardsToShow);
      const newCards = rawCards.map(rawCard => ({
        ...rawCard,
        word_dictionary: rawCard.word_dictionary 
          ? Object.fromEntries(
              Object.entries(rawCard.word_dictionary).filter(([, value]) => value !== undefined)
            ) as Record<string, number>
          : undefined
      } as CardType)).reverse();
      
      setCards(newCards);
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

  // Add keyboard event listener for arrow keys
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!showCards) return; // Only allow swiping when cards are shown
      
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
  }, [showCards]); // Re-run effect when showCards changes

  return (
    <div className="min-h-screen flex flex-col bg-neutral-100">
      {/* Top Search Bar */}
      <div className="flex justify-center p-4 bg-neutral/50 backdrop-blur-sm border-b border-white/20">
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
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex justify-center items-start pt-4">
        {/* Disliked Cards - Left Side */}
        <div className="w-96 p-4 overflow-y-auto max-h-[calc(100vh-120px)]" style={{ direction: 'rtl' }}>
          <div style={{ direction: 'ltr' }}>
            {showCards && (
              <div className="flex justify-center mb-4">
                <img src={redx} alt="Disliked" className="h-8" />
              </div>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2 2xl:grid-cols-3 gap-4">
            <AnimatePresence>
              {dislikedCards.map((card) => (
                <motion.div
                  key={`disliked-${card.id}`}
                  className="break-inside-avoid"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                >
                  <div 
                    className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-lg transition-shadow duration-200"
                    onClick={() => window.open(card.url, '_blank')}
                  >
                    <img
                      src={card.url}
                      alt={card.title || "Disliked card"}
                      className="w-full h-32 object-cover"
                    />
                    <div className="p-3">
                      <h4 className="text-sm font-medium text-gray-800 truncate">
                        {card.title || "Untitled"}
                      </h4>
                      <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                        {card.description || card.url}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Main Content - Center */}
        <div className="w-96 mr-14">
          {showCards && (
            <div className="flex justify-center">
              <SwipeCards 
                ref={swipeCardsRef}
                key={key} 
                cards={cards} 
                setCards={setCards}
                onLike={(card) => setLikedCards(prev => [card, ...prev])}
                onDislike={(card) => setDislikedCards(prev => [card, ...prev])}
              />
            </div>
          )}
          
          {/* Action Buttons */}
          {showCards && (
            <div className="flex justify-center gap-32 mt-10 ml-14">
              <button 
                onClick={handleDislikeClick}
                className="w-20 h-20 rounded-full bg-white/90 backdrop-blur-sm border-2 border-red-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200"
              >
                <img src={leftarrow} alt="dislike button" className="w-12 opacity-70 pr-2" />
              </button>
              <button 
                onClick={handleLikeClick}
                className="w-20 h-20 rounded-full bg-white/90 backdrop-blur-sm border-2 border-green-200/70 shadow-lg flex items-center justify-center hover:shadow-xl hover:scale-105 transition-all duration-200"
              >
                <img src={rightarrow} alt="like button" className="w-12 opacity-70 pl-2" />
              </button>
            </div>
          )}
        </div>

        {/* Liked Cards - Right Side */}
        <div className="w-96 p-4 overflow-y-auto max-h-[calc(100vh-120px)]">
          {showCards && (
            <div className="flex justify-center mb-4">
              <img src={greenHeartImg} alt="Liked" className="h-8" />
            </div>
          )}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2 2xl:grid-cols-3 gap-4">
            <AnimatePresence>
              {likedCards.map((card) => (
                <motion.div
                  key={`liked-${card.id}`}
                  className="break-inside-avoid"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ duration: 0.3, ease: "easeOut" }}
                >
                  <div 
                    className="bg-white rounded-lg shadow-md overflow-hidden cursor-pointer hover:shadow-lg transition-shadow duration-200"
                    onClick={() => window.open(card.url, '_blank')}
                  >
                    <img
                      src={card.url}
                      alt={card.title || "Liked card"}
                      className="w-full h-32 object-cover"
                    />
                    <div className="p-3">
                      <h4 className="text-sm font-medium text-gray-800 truncate">
                        {card.title || "Untitled"}
                      </h4>
                      <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                        {card.description || card.url}
                      </p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
