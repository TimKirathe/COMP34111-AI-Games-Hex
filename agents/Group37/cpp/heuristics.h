#ifndef HEURISTICS_H
#define HEURISTICS_H

#include "board.h"
#include <vector>
#include <utility>
#include <unordered_set>
#include <cstdint>

namespace hex {

// Two-bridge pattern structure
struct TwoBridgePattern {
    int dx1, dy1;  // First stone offset
    int dx2, dy2;  // Second stone offset
    int ex, ey;    // Empty cell offset
};

// Strong and weak opening positions for swap rule
extern const std::unordered_set<int> STRONG_OPENINGS;
extern const std::unordered_set<int> WEAK_OPENINGS;

// Helper: convert (x, y) to unique integer for set lookup
inline int position_to_key(int x, int y) {
    return x * BOARD_SIZE + y;
}

// Swap decision
bool should_swap(int opp_move_x, int opp_move_y, int board_size = BOARD_SIZE);

// Move priority scoring
double get_move_priority(const Board& board, std::pair<int, int> move, Colour colour);

// Weighted move selection
std::pair<int, int> select_weighted_move(
    const std::vector<std::pair<int, int>>& empty_tiles,
    const Board& board,
    Colour colour
);

// Shortest path distance using Dijkstra
int shortest_path_distance(const Board& board, Colour colour);

// Evaluate position (difference in shortest paths)
double evaluate_position(const Board& board, Colour colour);

// Zobrist hashing
uint64_t compute_board_hash(const Board& board);
uint64_t update_hash(uint64_t current_hash, int x, int y, Colour colour);

// Initialize Zobrist table (call once at startup)
void init_zobrist();

} // namespace hex

#endif // HEURISTICS_H
