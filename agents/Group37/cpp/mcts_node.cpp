#include "mcts_node.h"
#include "heuristics.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace hex {

MCTSNode::MCTSNode(const Board& board, Colour player_to_move,
                   MCTSNode* parent, std::pair<int, int> move,
                   uint64_t hash)
    : visits(0),
      wins(0.0),
      parent(parent),
      board(board),
      player_to_move(player_to_move),
      move(move),
      zobrist_hash_(hash),
      untried_moves(get_empty_tiles()) {
}

std::vector<std::pair<int, int>> MCTSNode::get_empty_tiles() const {
    return board.get_empty_tiles();
}

int MCTSNode::check_winner() const {
    return board.check_winner();
}

bool MCTSNode::is_fully_expanded() const {
    return untried_moves.empty();
}

bool MCTSNode::is_terminal() const {
    return check_winner() != 0 || get_empty_tiles().empty();
}

double MCTSNode::ucb1(double exploration_constant) const {
    if (visits == 0) {
        return std::numeric_limits<double>::infinity();
    }

    double exploitation = wins / visits;
    double exploration = exploration_constant * std::sqrt(std::log(parent->visits) / visits);

    return exploitation + exploration;
}

MCTSNode* MCTSNode::best_child(double exploration_constant) const {
    MCTSNode* best = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();

    for (const auto& [move, child] : children) {
        double value = child->ucb1(exploration_constant);
        if (value > best_value) {
            best_value = value;
            best = child.get();
        }
    }

    return best;
}

MCTSNode* MCTSNode::expand(std::pair<int, int> preferred_move) {
    if (untried_moves.empty()) {
        return nullptr;
    }

    // Select move with preferred_move priority
    std::pair<int, int> move;
    if (preferred_move.first != -1 && preferred_move.second != -1) {
        // Check if preferred_move is in untried_moves
        auto it = std::find(untried_moves.begin(), untried_moves.end(),
                           preferred_move);
        if (it != untried_moves.end()) {
            move = preferred_move;
            untried_moves.erase(it);
        } else {
            // Preferred move not available, use default
            move = untried_moves.back();
            untried_moves.pop_back();
        }
    } else {
        // No preferred move, use default
        move = untried_moves.back();
        untried_moves.pop_back();
    }

    // Create new board with the move applied
    Board new_board = board;
    new_board.set_tile(move.first, move.second, player_to_move);

    // Compute child hash incrementally
    uint64_t child_hash;
    if (zobrist_hash_ != 0) {
        // Parent has valid hash, update incrementally
        child_hash = update_hash(zobrist_hash_, move.first, move.second, player_to_move);
    } else {
        // Parent hash is zero, compute from scratch
        child_hash = compute_board_hash(new_board);
    }

    // Next player
    Colour next_player = opposite_colour(player_to_move);

    // Create child node
    auto child = std::make_unique<MCTSNode>(new_board, next_player, this, move, child_hash);
    MCTSNode* child_ptr = child.get();

    // Add to children
    children[move] = std::move(child);

    return child_ptr;
}

} // namespace hex
