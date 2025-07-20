import React, { forwardRef, useImperativeHandle } from "react";
import { motion, useMotionValue, useTransform } from "framer-motion";
import missingUpImg from "../assets/img/missing_up.png";

type CardType = {
  id: number; // rank of the card
  url: string;
  favicon?: string; // Optional favicon property
  title?: string; // Optional title property
  description?: string; // Optional description property
  image?: string; // Optional image property
  word_dictionary?: Record<string, number>; // Optional word dictionary for translations
};

export { type CardType };

type SwipeCardsProps = {
  cards: CardType[];
  setCards: React.Dispatch<React.SetStateAction<CardType[]>>;
  onLike: (card: CardType) => void;
  onDislike: (card: CardType) => void;
  autoOpenPages: boolean;
};

export type SwipeCardsRef = {
  swipeLeft: () => void;
  swipeRight: () => void;
};

const SwipeCards = forwardRef<SwipeCardsRef, SwipeCardsProps>(({ cards, setCards, onLike, onDislike, autoOpenPages}, ref) => {
  const animateSwipe = (direction: 'left' | 'right', topCard: CardType) => {
    const cardElement = document.querySelector(`[data-card-id="${topCard.id}"]`) as HTMLElement;
    if (cardElement) {
      const distance = direction === 'left' ? -150 : 150;
      const rotation = direction === 'left' ? -15 : 15;
      
      // Animate the card out
      cardElement.style.transform = `translateX(${distance}px) rotate(${rotation}deg)`;
      cardElement.style.opacity = '0.3';
      cardElement.style.transition = 'transform 0.3s ease-out, opacity 0.3s ease-out';
      
      // Remove card after animation
      setTimeout(() => {
        if (direction === 'left') {
          onDislike(topCard);
        } else {
            // Only open the page automatically if autoOpenPages is true
            if (autoOpenPages) {
              window.open(topCard.url, '_blank');
            }
          onLike(topCard);
        }
        setCards((prev) => prev.filter((card) => card.id !== topCard.id));
      }, 300);
    } else {
      // Fallback if element not found
      if (direction === 'left') {
        onDislike(topCard);
      } else {
        if (autoOpenPages) {
          window.open(topCard.url, '_blank');
        }
        onLike(topCard);
      }
      setCards((prev) => prev.filter((card) => card.id !== topCard.id));
    }
  };

  const swipeLeft = () => {
    if (cards.length > 0) {
      const topCard = cards[cards.length - 1];
      animateSwipe('left', topCard);
    }
  };

  const swipeRight = () => {
    if (cards.length > 0) {
      const topCard = cards[cards.length - 1];
      animateSwipe('right', topCard);
    }
  };

  useImperativeHandle(ref, () => ({
    swipeLeft,
    swipeRight
  }));

  return (
    <div
      className="relative h-[500px] w-[300px] grid place-items-center"

    >
      {cards.map((card, index) => (
        <Card
          key={card.id}
          {...card}
          cards={cards}
          setCards={setCards}
          onLike={onLike}
          onDislike={onDislike}
          index={index}
          autoOpenPages={autoOpenPages}
        />
      ))}
    </div>
  );
});

type CardProps = CardType & {
  setCards: React.Dispatch<React.SetStateAction<CardType[]>>;
  cards: CardType[];
  onLike: (card: CardType) => void;
  onDislike: (card: CardType) => void;
  index: number;
  autoOpenPages: boolean;
};

const Card = ({ id, url, favicon, title, description, image, word_dictionary, setCards, cards, onLike, onDislike, index, autoOpenPages }: CardProps) => {
  const x = useMotionValue(0);
  const rotateRaw = useTransform(x, [-150, 150], [-18, 18]);
  const opacity = useTransform(x, [-150, 0, 150], [0.2, 1, 0.2]);

  const isFront = id === cards[cards.length - 1].id;

  const rotate = useTransform(() => {
    const offset = isFront ? 0 : id % 2 ? 2 : -2; // Reduced rotation offset
    return `${rotateRaw.get() + offset}deg`;
  });

  const handleDragEnd = () => {
    const currentCard = { id, url, favicon, title, description, image, word_dictionary };
    
    if (x.get() > 100 ){
      if (autoOpenPages) {
        open(url, '_blank');
      }
      onLike(currentCard);
    } else if (x.get() < -100) { 
      onDislike(currentCard);
    }

    if (Math.abs(x.get()) > 100) {
      setCards((prev) => prev.filter((card) => card.id !== id));
    }
  };

  return (
       <motion.div
      data-card-id={id}
      className=" origin-bottom rounded-lg overflow-hidden relative flex flex-col justify-end bg-white shadow-lg hover:cursor-grab active:cursor-grabbing"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        x,
        opacity,
        rotate,
        transition: "0.125s transform",
        height: "32rem",
        width: "22rem"

      }}
      animate={{
        y: 0,
        opacity: 1,
        scale: isFront ? 1 : 0.98,
      }}
      initial={{ y: 0, opacity: 0 }}
      transition={{
        duration: 0.3,
        delay: Math.max(0, (cards.length - index - 1) * 0.1), // Reverse delay order
        type: "tween",
        ease: "easeOut"
      }}
      drag={isFront ? true : false}
   
      dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
      onDragEnd={handleDragEnd}
    >
      {/* Background image */}
      <img
        src={image || url || missingUpImg}
        alt="Card background"
        className="absolute inset-0 h-full w-full object-cover z-0 pointer-events-none"
        style={{
          objectPosition: (image || url) ? 'center top' : 'center'
        }}
        onError={(e) => {
          console.log('Image failed to load:', image || url);
          // If main image fails, try the URL as fallback, then missing image
          if (image && e.currentTarget.src === image) {
            e.currentTarget.src = url;
          } else if (e.currentTarget.src === url) {
            e.currentTarget.src = missingUpImg;
            // Use center position for missing image
            e.currentTarget.style.objectPosition = 'center';
          }
        }}
        onLoad={() => {
          console.log('Image loaded successfully:', image || url);
        }}
      />

      {/* Word dictionary features at the top */}
      {word_dictionary && Object.keys(word_dictionary).length > 0 && (
        <div className="relative z-10 text-white p-3 ">
          <div className="flex flex-wrap gap-2">
            {Object.entries(word_dictionary).map(([word, score]) => {
              // Map score (0.0-1.0) to orange color scale (50-950)
              const getOrangeClass = (score: number) => {
                if (score <= 0) return 'bg-transparent border-gray-300 text-gray-400';
                
                const clampedScore = Math.max(0, Math.min(1, score));
                const colorStep = Math.round(clampedScore * 9); // 0-9 range
                
                const orangeClasses = [
                  'bg-blue-600 border-white text-white',   // 0.0-0.1
                  'bg-blue-500 border-white text-white',  // 0.1-0.2
                  'bg-blue-400 border-white text-white',  // 0.2-0.3
                  'bg-blue-300 border-white text-white',  // 0.3-0.4
                  'bg-blue-100 border-white text-white',  // 0.4-0.5
                  'bg-orange-300 border-white text-white',       // 0.5-0.6
                  'bg-orange-500 border-white text-white',       // 0.6-0.7
                  'bg-orange-700 border-white text-white',       // 0.7-0.8
                  'bg-red-700 border-white text-white',       // 0.8-0.9
                  'bg-red-800 border-white text-white'        // 0.9-1.0
                ];
                
                return orangeClasses[colorStep];
              };
              
              return (
                <span
                  key={word}
                  className={`inline-flex items-center px-2 py-1 rounded-full text-xs border ${getOrangeClass(score as number)}`}
                  title={`Score: ${score}`}
                >
                  {(score as number) >= 0.5 ? '🔥' : '🧊'}
                  <span className="ml-1">{word}</span>
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Spacer to push content to bottom */}
      <div className="flex-1"></div>

      {/* Overlay content */}
      <div className="relative z-10 bg-black/40 text-white p-4 backdrop-blur-sm">
        
        <div className="flex items-center gap-2">
          {favicon && (
            <img
              src={favicon}
              alt="favicon"
              className="w-5 h-5"
            />
          )}
          <h2 className="text-lg font-semibold">
            {title || "Untitled"}
          </h2>
        </div>
        <div> 
          <p className="text-xs pb-2">{url}</p>
        </div>
        <p className="text-sm">{description || url }</p>
      
      </div>
    </motion.div>
  );
};

export default SwipeCards;
