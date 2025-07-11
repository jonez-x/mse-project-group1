import { motion, AnimatePresence } from "framer-motion";
import SwipeCards, { type CardType, type SwipeCardsRef } from "./SwipeCards";
import { type RefObject } from "react";

interface TinderLikeViewProps {
  cards: CardType[];
  setCards: React.Dispatch<React.SetStateAction<CardType[]>>;
  likedCards: CardType[];
  dislikedCards: CardType[];
  onLike: (card: CardType) => void;
  onDislike: (card: CardType) => void;
  swipeCardsRef: RefObject<SwipeCardsRef | null>;
  componentKey: number;
}

const TinderLikeView = ({ 
  cards, 
  setCards, 
  likedCards, 
  dislikedCards, 
  onLike, 
  onDislike, 
  swipeCardsRef, 
  componentKey 
}: TinderLikeViewProps) => {
  return (
    <div className="flex-1 flex justify-center items-start pt-4">
      {/* Disliked Cards - Left Side */}
      <div className="w-96 p-4 overflow-y-auto max-h-[calc(100vh-200px)]" style={{ direction: 'rtl', width: '650px' }}>
        <div style={{ direction: 'ltr' }}>
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
        <div className="flex justify-center">
          <SwipeCards 
            ref={swipeCardsRef}
            key={componentKey} 
            cards={cards} 
            setCards={setCards}
            onLike={onLike}
            onDislike={onDislike}
          />
        </div>
      </div>

      {/* Liked Cards - Right Side */}
      <div className="p-4 overflow-y-auto max-h-[calc(100vh-200px)]" style={{ width: '650px' }}>
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
  );
};

export default TinderLikeView;
