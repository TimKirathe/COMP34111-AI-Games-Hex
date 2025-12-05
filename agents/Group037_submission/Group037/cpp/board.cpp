#include "board.h"
#include <algorithm>
#include <stack>

namespace hex {

Board::Board() {
    tiles_.fill(Tile::EMPTY);
}

Board::Board(const Board& other) : tiles_(other.tiles_) {}

Board& Board::operator=(const Board& other) {
    if (this != &other) {
        tiles_ = other.tiles_;
    }
    return *this;
}

Tile Board::get_tile(int x, int y) const {
    return tiles_[index(x, y)];
}

void Board::set_tile(int x, int y, Colour colour) {
    tiles_[index(x, y)] = (colour == Colour::RED) ? Tile::RED : Tile::BLUE;
}

void Board::clear_tile(int x, int y) {
    tiles_[index(x, y)] = Tile::EMPTY;
}

bool Board::is_valid_position(int x, int y) const {
    return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE;
}

std::vector<std::pair<int, int>> Board::get_neighbors(int x, int y) const {
    std::vector<std::pair<int, int>> neighbors;
    neighbors.reserve(NEIGHBOUR_COUNT);

    for (int k = 0; k < NEIGHBOUR_COUNT; ++k) {
        int nx = x + I_DISPLACEMENTS[k];
        int ny = y + J_DISPLACEMENTS[k];
        if (is_valid_position(nx, ny)) {
            neighbors.emplace_back(nx, ny);
        }
    }

    return neighbors;
}

std::vector<std::pair<int, int>> Board::get_empty_tiles() const {
    std::vector<std::pair<int, int>> empty;
    empty.reserve(TOTAL_TILES);

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            if (tiles_[index(i, j)] == Tile::EMPTY) {
                empty.emplace_back(i, j);
            }
        }
    }

    return empty;
}

void Board::from_flat_array(const std::vector<int>& arr) {
    for (size_t i = 0; i < arr.size() && i < TOTAL_TILES; ++i) {
        tiles_[i] = static_cast<Tile>(arr[i]);
    }
}

std::vector<int> Board::to_flat_array() const {
    std::vector<int> arr(TOTAL_TILES);
    for (int i = 0; i < TOTAL_TILES; ++i) {
        arr[i] = static_cast<int>(tiles_[i]);
    }
    return arr;
}

// Check for winner using iterative DFS
// Returns 0 if no winner, 1 for RED, 2 for BLUE
int Board::check_winner() const {
    std::vector<bool> visited(TOTAL_TILES, false);

    // Check RED (top to bottom: row 0 to row BOARD_SIZE-1)
    for (int j = 0; j < BOARD_SIZE; ++j) {
        if (tiles_[index(0, j)] == Tile::RED && !visited[index(0, j)]) {
            if (dfs_win(0, j, Colour::RED, visited)) {
                return 1; // RED wins
            }
        }
    }

    // Check BLUE (left to right: col 0 to col BOARD_SIZE-1)
    visited.assign(TOTAL_TILES, false);
    for (int i = 0; i < BOARD_SIZE; ++i) {
        if (tiles_[index(i, 0)] == Tile::BLUE && !visited[index(i, 0)]) {
            if (dfs_win(i, 0, Colour::BLUE, visited)) {
                return 2; // BLUE wins
            }
        }
    }

    return 0; // No winner
}

bool Board::dfs_win(int x, int y, Colour colour, std::vector<bool>& visited) const {
    // Use iterative DFS to avoid stack overflow
    std::stack<std::pair<int, int>> stack;
    stack.push({x, y});

    Tile target_tile = (colour == Colour::RED) ? Tile::RED : Tile::BLUE;

    while (!stack.empty()) {
        auto [cx, cy] = stack.top();
        stack.pop();

        int idx = index(cx, cy);
        if (visited[idx]) {
            continue;
        }
        visited[idx] = true;

        // Goal check
        if (colour == Colour::RED && cx == BOARD_SIZE - 1) {
            return true;
        }
        if (colour == Colour::BLUE && cy == BOARD_SIZE - 1) {
            return true;
        }

        // Explore neighbors
        for (int k = 0; k < NEIGHBOUR_COUNT; ++k) {
            int nx = cx + I_DISPLACEMENTS[k];
            int ny = cy + J_DISPLACEMENTS[k];

            if (is_valid_position(nx, ny)) {
                int nidx = index(nx, ny);
                if (!visited[nidx] && tiles_[nidx] == target_tile) {
                    stack.push({nx, ny});
                }
            }
        }
    }

    return false;
}

} // namespace hex
