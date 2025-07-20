import { motion, AnimatePresence } from "framer-motion";
import { type CardType } from "./SwipeCards";
import { useState, useMemo } from "react";

interface ClassicListViewProps {
  cards: CardType[];
  searchQuery: string;
}

const ClassicListView = ({ cards }: ClassicListViewProps) => {
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 20;

  const filteredAndSortedCards = useMemo(() => {
    // For now, don't filter by search query - this will be handled by backend
    // Just sort all cards by ID (lowest first)
    return cards.sort((a, b) => {
      const idA = typeof a.id === 'string' ? parseInt(a.id) || 0 : a.id || 0;
      const idB = typeof b.id === 'string' ? parseInt(b.id) || 0 : b.id || 0;
      return idA - idB;
    });
  }, [cards]); // Removed searchQuery dependency

  const totalPages = Math.ceil(filteredAndSortedCards.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentCards = filteredAndSortedCards.slice(startIndex, endIndex);

  // Reset to first page when search changes (disabled for now)
  // useEffect(() => {
  //   setCurrentPage(1);
  // }, [searchQuery]);

  const goToPage = (page: number) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">All Results</h2>
        <p className="text-gray-600">
          Showing {startIndex + 1}-{Math.min(endIndex, filteredAndSortedCards.length)} of {filteredAndSortedCards.length} 
          {totalPages > 1 && (
            <span className="ml-2">
              (Page {currentPage} of {totalPages})
            </span>
          )}
        </p>
      </div>
      
      <div className="space-y-4">
        <AnimatePresence>
          {currentCards.map((card, index) => (
            <motion.div
              key={card.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className="  hover:shadow-lg transition-shadow duration-200 overflow-hidden"
            >
              <div className="flex items-start p-2">
                <div className="flex-shrink-0 mr-4">
                     <img src={card.favicon} alt="" className="h-4 inline-block"/>
                </div>
                {/* Card Content */}
                <div className="flex-1 min-w-0">
                  <h3 
                    className="text-lg font-semibold text-gray-900 cursor-pointer hover:text-blue-600 transition-colors"
                    onClick={() => window.open(card.url, '_blank')}
                  >
                    {card.title || "Untitled"}
                  </h3>
                   {/* Favicon and Domain */}
                  <div className="flex items-center gap-2 ">
                    <span 
                      className="text-sm text-gray-500 hover:text-blue-600 cursor-pointer transition-colors"
                      onClick={() => window.open(card.url, '_blank')}
                    >
                      {new URL(card.url).hostname}
                    </span>
                  </div>
                  <p className="text-gray-600 mb-3 line-clamp-2">
                    {card.description || card.url}
                  </p>
                  
                 
                  
                  {/* Term Frequency Heat Map Tags */}
                  {card.word_dictionary && Object.keys(card.word_dictionary).length > 0 && card.document_length && card.document_length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {Object.entries(card.word_dictionary)
                        .map(([word, rawCount]) => {
                          // Calculate Term Frequency (TF) = word_count / document_length
                          const tfScore = rawCount / card.document_length!;
                          
                          // Calculate all TF scores for normalization
                          const allTfScores = Object.entries(card.word_dictionary!).map(([, count]) => count / card.document_length!);
                          const maxTf = Math.max(...allTfScores);
                          const minTf = Math.min(...allTfScores);
                          
                          // Normalize TF score to 0.1-1.0 range for better visualization
                          let normalizedScore = 0.1; // Minimum visibility
                          if (maxTf > minTf) {
                            normalizedScore = 0.1 + 0.9 * ((tfScore - minTf) / (maxTf - minTf));
                          } else if (tfScore > 0) {
                            normalizedScore = 0.5; // All words have same frequency
                          }
                          
                          return { word, rawCount, tfScore, normalizedScore };
                        })
                        .sort((a, b) => b.normalizedScore - a.normalizedScore) // Sort by normalized score
                        .slice(0, 5) // Show top 5 terms
                        .map(({ word, rawCount, tfScore, normalizedScore }) => {
                          // Map score (0.1-1.0) to color scale (same as SwipeCards)
                          const getColorClass = (score: number) => {
                            if (score <= 0) return 'bg-transparent border-gray-300 text-gray-400';
                            
                            const clampedScore = Math.max(0, Math.min(1, score));
                            const colorStep = Math.round(clampedScore * 9); // 0-9 range
                            
                            const colorClasses = [
                              'bg-blue-600 border-blue-600 text-white',   // 0.0-0.1
                              'bg-blue-500 border-blue-500 text-white',   // 0.1-0.2
                              'bg-blue-400 border-blue-400 text-white',   // 0.2-0.3
                              'bg-blue-300 border-blue-300 text-white',   // 0.3-0.4
                              'bg-blue-100 border-blue-100 text-gray-800', // 0.4-0.5
                              'bg-orange-300 border-orange-300 text-white', // 0.5-0.6
                              'bg-orange-500 border-orange-500 text-white', // 0.6-0.7
                              'bg-orange-700 border-orange-700 text-white', // 0.7-0.8
                              'bg-red-700 border-red-700 text-white',      // 0.8-0.9
                              'bg-red-800 border-red-800 text-white'       // 0.9-1.0
                            ];
                            
                            return colorClasses[colorStep];
                          };
                          
                          return (
                            <span
                              key={word}
                              className={`inline-flex items-center px-2 py-1 rounded-full text-xs border ${getColorClass(normalizedScore)}`}
                              title={`Raw: ${rawCount}, TF: ${tfScore.toFixed(4)}, Score: ${normalizedScore.toFixed(2)}`}
                            >
                              {normalizedScore >= 0.5 ? '🔥' : '🧊'}
                              <span className="ml-1">{word}</span>
                            </span>
                          );
                        })}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {filteredAndSortedCards.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-400 text-6xl mb-4">🔍</div>
            <h3 className="text-xl font-medium text-gray-900 mb-2">No results found</h3>
            <p className="text-gray-600">Try adjusting your search terms</p>
          </div>
        )}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="mt-8 flex justify-center items-center space-x-2">
          <button
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage === 1}
            className="px-3 py-2 rounded-md bg-white border border-gray-300 text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            let pageNum;
            if (totalPages <= 5) {
              pageNum = i + 1;
            } else if (currentPage <= 3) {
              pageNum = i + 1;
            } else if (currentPage >= totalPages - 2) {
              pageNum = totalPages - 4 + i;
            } else {
              pageNum = currentPage - 2 + i;
            }
            
            return (
              <button
                key={pageNum}
                onClick={() => goToPage(pageNum)}
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  currentPage === pageNum
                    ? 'bg-blue-600 text-white'
                    : 'bg-white border border-gray-300 text-gray-500 hover:bg-gray-50'
                }`}
              >
                {pageNum}
              </button>
            );
          })}
          
          <button
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="px-3 py-2 rounded-md bg-white border border-gray-300 text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default ClassicListView;
