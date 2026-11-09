#ifndef RESULT_RANKER_HPP
#define RESULT_RANKER_HPP

#include "Document.hpp"
#include "HashMap.hpp"
#include "InvertedIndex.hpp"

#include <vector>
#include <string>
#include <utility>

class ResultRanker {

public:

    // Rank documents using TF-IDF
    std::vector<std::pair<int, double>> rank(
        const std::vector<std::string>& queryTerms,
        const InvertedIndex& index,
        int totalDocuments
    ) const;


    // Return Top K relevant documents
    std::vector<std::pair<int, double>> rankTopK(
        const std::vector<std::string>& queryTerms,
        const InvertedIndex& index,
        int totalDocuments,
        int k
    ) const;

};

#endif