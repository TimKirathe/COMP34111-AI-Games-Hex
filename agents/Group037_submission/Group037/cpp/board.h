#ifndef BOARD_H
#define BOARD_H

#include <array>
#include <vector>
#include <utility>
#include <cstdint>

namespace hex {

// Tile states
enum class Tile : int8_t {
    EMPTY = 0,
    RED = 1,
    BLUE = 2
};

// Color type for compatibility
enum class Colour : int8_t {
    RED = 1,
    BLUE = 2
};

// Board size (fixed at 11x11 for Hex)
constexpr int BOARD_SIZE = 11;
constexpr int TOTAL_TILES = BOARD_SIZE * BOARD_SIZE;

// Neighbor displacements (6 neighbors in hexagonal grid)
// From Tile.py: I_DISPLACEMENTS = [-1, -1, 0, 1, 1, 0]
//               J_DISPLACEMENTS = [0, 1, 1, 0, -1, -1]
constexpr int NEIGHBOUR_COUNT = 6;
constexpr int I_DISPLACEMENTS[NEIGHBOUR_COUNT] = {-1, -1, 0, 1, 1, 0};
constexpr int J_DISPLACEMENTS[NEIGHBOUR_COUNT] = {0, 1, 1, 0, -1, -1};

// Helper function
inline Colour opposite_colour(Colour c) {
    return (c == Colour::RED) ? Colour::BLUE : Colour::RED;
}

class Board {
public:
    Board();

    // Copy constructor for efficient board cloning
    Board(const Board& other);

    // Assignment operator
    Board& operator=(const Board& other);

    // Get tile at position
    Tile get_tile(int x, int y) const;

    // Set tile at position
    void set_tile(int x, int y, Colour colour);

    // Clear tile at position (set to EMPTY)
    void clear_tile(int x, int y);

    // Get valid neighbors for a position
    std::vector<std::pair<int, int>> get_neighbors(int x, int y) const;

    // Check if position is valid
    bool is_valid_position(int x, int y) const;

    // Get all empty tile positions
    std::vector<std::pair<int, int>> get_empty_tiles() const;

    // Check for winner (returns Colour or nullopt if no winner)
    // Returns 0 if no winner, 1 for RED, 2 for BLUE
    int check_winner() const;

    // Get board size
    int size() const { return BOARD_SIZE; }

    // Convert to/from flat array representation for Python interface
    void from_flat_array(const std::vector<int>& arr);
    std::vector<int> to_flat_array() const;

private:
    // Board state: flat array for cache efficiency
    std::array<Tile, TOTAL_TILES> tiles_;

    // Helper for index calculation
    inline int index(int x, int y) const {
        return x * BOARD_SIZE + y;
    }

    // DFS-based win detection
    bool dfs_win(int x, int y, Colour colour, std::vector<bool>& visited) const;
};

} // namespace hex

#endif // BOARD_H
