#include "ResultRanker.hpp"

#include <algorithm>
#include <cmath>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace std;


// Helper function to calculate TF-IDF scores

namespace {

unordered_map<int, double> calculateScores(
    const vector<string>& queryTerms,
    const InvertedIndex& index,
    int totalDocuments
) {

    unordered_map<int, double> scores;

    unordered_set<string> processedWords;

    // Process each query word
    for (const string& word : queryTerms) {

        // Avoid duplicate words
        if (!processedWords.insert(word).second)
            continue;

        // Get posting list from Inverted Index
        const PostingNode* result = index.find(word);

        if (result == nullptr)
            continue;


        // Calculate Document Frequency (DF)

        int df = 0;

        for (const PostingNode* temp = result;
             temp != nullptr;
             temp = temp->next) {

            df++;
        }


        // Calculate Inverse Document Frequency (IDF)

        double idf = log(
            (double)(totalDocuments + 1) / (df + 1)
        ) + 1.0;


        // Calculate TF-IDF for each document

        for (const PostingNode* temp = result;
             temp != nullptr;
             temp = temp->next) {

            int tf = temp->frequency;

            double tfidf = tf * idf;

            scores[temp->documentID] += tfidf;
        }
    }

    return scores;
}

}


// Rank all documents using TF-IDF

vector<pair<int, double>> ResultRanker::rank(
    const vector<string>& queryTerms,
    const InvertedIndex& index,
    int totalDocuments
) const {

    if (totalDocuments <= 0)
        return {};

    // Calculate document scores
    auto scores = calculateScores(
        queryTerms,
        index,
        totalDocuments
    );


    // Convert scores into vector

    vector<pair<int, double>> rankedResult(
        scores.begin(),
        scores.end()
    );


    // Sort according to relevance score

    sort(
        rankedResult.begin(),
        rankedResult.end(),

        [](const pair<int, double>& a,
           const pair<int, double>& b) {

            if (a.second == b.second)
                return a.first < b.first;

            return a.second > b.second;
        }
    );

    return rankedResult;
}



// Return Top K results using Priority Queue

vector<pair<int, double>> ResultRanker::rankTopK(
    const vector<string>& queryTerms,
    const InvertedIndex& index,
    int totalDocuments,
    int k
) const {

    if (k <= 0 || totalDocuments <= 0)
        return {};


    // Calculate TF-IDF scores

    auto scores = calculateScores(
        queryTerms,
        index,
        totalDocuments
    );


    // Create Min Heap

    auto compare = [](const pair<int, double>& a,
                      const pair<int, double>& b) {

        if (a.second == b.second)
            return a.first < b.first;

        return a.second > b.second;
    };


    priority_queue<
        pair<int, double>,
        vector<pair<int, double>>,
        decltype(compare)
    > pq(compare);


    // Maintain only K best results

    for (const auto& result : scores) {

        pq.push(result);

        if (pq.size() > static_cast<size_t>(k)) {
            pq.pop();
        }
    }


    // Extract Top K results

    vector<pair<int, double>> topResults;

    while (!pq.empty()) {

        topResults.push_back(pq.top());

        pq.pop();
    }


    // Display highest score first

    reverse(topResults.begin(), topResults.end());

    return topResults;
}