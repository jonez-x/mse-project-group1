import React from "react";
import { motion, useMotionValue, useTransform } from "framer-motion";

type CardType = {
  id: number; // rank of the card
  url: string;
  favicon?: string; // Optional favicon property
  title?: string; // Optional title property
  description?: string; // Optional description property
  word_dictionary?: Record<string, string>; // Optional word dictionary for translations
  
};

export { type CardType };

type SwipeCardsProps = {
  cards: CardType[];
  setCards: React.Dispatch<React.SetStateAction<CardType[]>>;
};

const SwipeCards = ({ cards, setCards }: SwipeCardsProps) => {
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
          index={index}
        />
      ))}
    </div>
  );
};

type CardProps = CardType & {
  setCards: React.Dispatch<React.SetStateAction<CardType[]>>;
  cards: CardType[];
  index: number;
};

const Card = ({ id, url, favicon, title, description, setCards, cards, index }: CardProps) => {
  const x = useMotionValue(0);
  const rotateRaw = useTransform(x, [-150, 150], [-18, 18]);
  const opacity = useTransform(x, [-150, 0, 150], [0, 1, 0]);

  const isFront = id === cards[cards.length - 1].id;

  const rotate = useTransform(() => {
    
    const offset = isFront ? 0 : id % 2 ? 2 : -2; // Reduced rotation offset
    return `${rotateRaw.get() + offset}deg`;
  });

  const handleDragEnd = () => {
    if (Math.abs(x.get()) > 100) {
      setCards((prev) => prev.filter((card) => card.id !== id));
    }
  };

  return (
       <motion.div
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
        src={url}
        alt="Card background"
        className="absolute inset-0 h-full w-full object-cover z-0 pointer-events-none"
      />

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
            {title + `Card #${id}`}
          </h2>
        </div>
        <p className="text-sm">{description || url }</p>
        <button className="mt-2 px-3 py-1 bg-blue-500 rounded text-sm hover:bg-blue-600">
          Action
        </button>
      </div>
    </motion.div>
  );
};

export default SwipeCards;
