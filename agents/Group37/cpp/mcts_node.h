#ifndef MCTS_NODE_H
#define MCTS_NODE_H

#include "board.h"
#include <map>
#include <memory>
#include <vector>
#include <utility>

namespace hex {

class MCTSNode {
public:
    // Constructor
    MCTSNode(const Board& board, Colour player_to_move,
             MCTSNode* parent = nullptr,
             std::pair<int, int> move = {-1, -1},
             uint64_t hash = 0);

    // Statistics
    int visits;
    double wins;

    // Tree structure
    MCTSNode* parent;
    std::map<std::pair<int, int>, std::unique_ptr<MCTSNode>> children;

    // Game state
    Board board;
    Colour player_to_move;
    std::pair<int, int> move; // Move that led to this node
    uint64_t zobrist_hash_;

    // Untried moves
    std::vector<std::pair<int, int>> untried_moves;

    // Get Zobrist hash
    uint64_t get_hash() const { return zobrist_hash_; }

    // UCB1 calculation
    double ucb1(double exploration_constant = 1.414) const;

    // Get best child according to UCB1
    MCTSNode* best_child(double exploration_constant = 1.414) const;

    // Check if node is fully expanded
    bool is_fully_expanded() const;

    // Check if node is terminal
    bool is_terminal() const;

    // Expand: add a new child node for an untried move
    MCTSNode* expand(std::pair<int, int> preferred_move = {-1, -1});

private:
    // Helper: Get all empty tiles on the board
    std::vector<std::pair<int, int>> get_empty_tiles() const;

    // Check if there's a winner
    int check_winner() const;
};

} // namespace hex

#endif // MCTS_NODE_H
