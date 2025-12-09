#ifndef MCTS_H
#define MCTS_H

#include "board.h"
#include "mcts_node.h"
#include "transposition_table.h"
#include <utility>
#include <vector>

namespace hex {

// Main MCTS search function
// Returns the best move as (x, y) coordinates
std::pair<int, int> search(
    const Board& board_state,
    Colour current_player,
    double time_limit
);

// Helper: Simulate a game from a node with heuristic guidance
int simulate_with_heuristics(MCTSNode* node, Colour agent_colour);

// Helper: Backpropagate result up the tree
void backpropagate(MCTSNode* node, int winner, Colour agent_colour,
                   TranspositionTable& tt);

// Helper: Find immediate winning move
std::pair<int, int> find_winning_move(const Board& board, Colour colour);

} // namespace hex

#endif // MCTS_H
