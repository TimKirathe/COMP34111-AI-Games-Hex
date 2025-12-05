#include "heuristics.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <queue>
#include <limits>
#include <unordered_map>

namespace hex {

// Strong openings (center and near-center positions)
// From Python: (5, 5), (4, 5), (5, 4), (4, 6), (6, 4), (5, 6), (6, 5),
//              (4, 4), (6, 6), (3, 5), (5, 3), (3, 6), (6, 3)
const std::unordered_set<int> STRONG_OPENINGS = {
    position_to_key(5, 5), position_to_key(4, 5), position_to_key(5, 4),
    position_to_key(4, 6), position_to_key(6, 4), position_to_key(5, 6),
    position_to_key(6, 5), position_to_key(4, 4), position_to_key(6, 6),
    position_to_key(3, 5), position_to_key(5, 3), position_to_key(3, 6),
    position_to_key(6, 3)
};

// Weak openings (edges and corners)
// From Python: (0, 0), (0, 10), (10, 0), (10, 10),
//              (0, 1), (1, 0), (0, 9), (9, 0), (1, 10), (10, 1), (9, 10), (10, 9)
const std::unordered_set<int> WEAK_OPENINGS = {
    position_to_key(0, 0), position_to_key(0, 10), position_to_key(10, 0), position_to_key(10, 10),
    position_to_key(0, 1), position_to_key(1, 0), position_to_key(0, 9), position_to_key(9, 0),
    position_to_key(1, 10), position_to_key(10, 1), position_to_key(9, 10), position_to_key(10, 9)
};

// Zobrist hash table
static std::unordered_map<int, uint64_t> ZOBRIST_TABLE;
static bool zobrist_initialized = false;

void init_zobrist() {
    if (zobrist_initialized) return;

    std::mt19937_64 rng(42); // Deterministic seed for reproducibility
    std::uniform_int_distribution<uint64_t> dist;

    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            for (int c = 1; c <= 2; ++c) { // RED=1, BLUE=2
                int key = i * BOARD_SIZE * 3 + j * 3 + c;
                ZOBRIST_TABLE[key] = dist(rng);
            }
        }
    }

    zobrist_initialized = true;
}

uint64_t compute_board_hash(const Board& board) {
    if (!zobrist_initialized) init_zobrist();

    uint64_t h = 0;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            Tile tile = board.get_tile(i, j);
            if (tile != Tile::EMPTY) {
                int colour_val = static_cast<int>(tile);
                int key = i * BOARD_SIZE * 3 + j * 3 + colour_val;
                h ^= ZOBRIST_TABLE[key];
            }
        }
    }
    return h;
}

uint64_t update_hash(uint64_t current_hash, int x, int y, Colour colour) {
    if (!zobrist_initialized) init_zobrist();

    int colour_val = static_cast<int>(colour);
    int key = x * BOARD_SIZE * 3 + y * 3 + colour_val;
    return current_hash ^ ZOBRIST_TABLE[key];
}

bool should_swap(int opp_move_x, int opp_move_y, int board_size) {
    int key = position_to_key(opp_move_x, opp_move_y);

    // Always swap strong openings
    if (STRONG_OPENINGS.count(key) > 0) {
        return true;
    }

    // Never swap weak openings
    if (WEAK_OPENINGS.count(key) > 0) {
        return false;
    }

    // For other moves, use distance from center heuristic
    int center = board_size / 2;
    int dist = std::abs(opp_move_x - center) + std::abs(opp_move_y - center);

    // Swap if within 2 tiles of center
    return dist <= 2;
}

double get_move_priority(const Board& board, std::pair<int, int> move, Colour colour) {
    int x = move.first;
    int y = move.second;
    double score = 0.0;

    // Bonus for extending existing connections
    auto neighbors = board.get_neighbors(x, y);
    for (const auto& [nx, ny] : neighbors) {
        Tile tile = board.get_tile(nx, ny);
        if (tile == (colour == Colour::RED ? Tile::RED : Tile::BLUE)) {
            score += 2.0;
        }
    }

    // Bonus for blocking opponent connections
    Colour opp = opposite_colour(colour);
    Tile opp_tile = (opp == Colour::RED ? Tile::RED : Tile::BLUE);
    for (const auto& [nx, ny] : neighbors) {
        if (board.get_tile(nx, ny) == opp_tile) {
            score += 1.5;
        }
    }

    // Bonus for center positions (strategic value)
    int center = BOARD_SIZE / 2;
    int dist_to_center = std::abs(x - center) + std::abs(y - center);
    score += std::max(0.0, (BOARD_SIZE - dist_to_center) / static_cast<double>(BOARD_SIZE));

    // Bonus for positions along main diagonal
    if (colour == Colour::RED) {
        // RED benefits from positions along y axis progression
        if (x == y) score += 0.5;
    } else {
        // BLUE benefits from positions along x axis progression
        if (x == y) score += 0.5;
    }

    return score;
}

std::pair<int, int> select_weighted_move(
    const std::vector<std::pair<int, int>>& empty_tiles,
    const Board& board,
    Colour colour
) {
    if (empty_tiles.empty()) {
        return {-1, -1};
    }

    if (empty_tiles.size() == 1) {
        return empty_tiles[0];
    }

    // For speed, only score a subset of moves
    int sample_size = std::min(8, static_cast<int>(empty_tiles.size()));

    static thread_local std::mt19937 gen(std::random_device{}());

    std::vector<std::pair<int, int>> candidates;
    if (static_cast<int>(empty_tiles.size()) <= sample_size) {
        candidates = empty_tiles;
    } else {
        // Random sample
        std::sample(empty_tiles.begin(), empty_tiles.end(),
                   std::back_inserter(candidates), sample_size, gen);
    }

    std::pair<int, int> best_move = candidates[0];
    double best_score = -std::numeric_limits<double>::infinity();

    std::uniform_real_distribution<> dis(0.0, 0.1);

    for (const auto& move : candidates) {
        double score = get_move_priority(board, move, colour);
        // Add small random noise to break ties
        score += dis(gen);

        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
    }

    return best_move;
}

int shortest_path_distance(const Board& board, Colour colour) {
    constexpr int INF = BOARD_SIZE * BOARD_SIZE + 1;

    std::vector<std::vector<int>> dist(BOARD_SIZE, std::vector<int>(BOARD_SIZE, INF));

    // Priority queue: (distance, x, y)
    using PQElement = std::tuple<int, int, int>;
    std::priority_queue<PQElement, std::vector<PQElement>, std::greater<>> pq;

    Tile my_tile = (colour == Colour::RED ? Tile::RED : Tile::BLUE);
    Tile opp_tile = (colour == Colour::RED ? Tile::BLUE : Tile::RED);

    // Initialize starting positions
    if (colour == Colour::RED) {
        // Start from top row
        for (int j = 0; j < BOARD_SIZE; ++j) {
            Tile tile = board.get_tile(0, j);
            if (tile == opp_tile) continue;

            int cost = (tile == my_tile) ? 0 : 1;
            dist[0][j] = cost;
            pq.push({cost, 0, j});
        }
    } else {
        // Start from left column
        for (int i = 0; i < BOARD_SIZE; ++i) {
            Tile tile = board.get_tile(i, 0);
            if (tile == opp_tile) continue;

            int cost = (tile == my_tile) ? 0 : 1;
            dist[i][0] = cost;
            pq.push({cost, i, 0});
        }
    }

    while (!pq.empty()) {
        auto [d, x, y] = pq.top();
        pq.pop();

        if (d > dist[x][y]) continue;

        // Check goal
        if (colour == Colour::RED && x == BOARD_SIZE - 1) {
            return d;
        }
        if (colour == Colour::BLUE && y == BOARD_SIZE - 1) {
            return d;
        }

        // Explore neighbors
        auto neighbors = board.get_neighbors(x, y);
        for (const auto& [nx, ny] : neighbors) {
            Tile tile = board.get_tile(nx, ny);
            if (tile == opp_tile) continue;

            int edge_cost = (tile == my_tile) ? 0 : 1;
            int new_dist = d + edge_cost;

            if (new_dist < dist[nx][ny]) {
                dist[nx][ny] = new_dist;
                pq.push({new_dist, nx, ny});
            }
        }
    }

    // Find minimum distance to goal edge
    int min_dist = INF;
    if (colour == Colour::RED) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            min_dist = std::min(min_dist, dist[BOARD_SIZE - 1][j]);
        }
    } else {
        for (int i = 0; i < BOARD_SIZE; ++i) {
            min_dist = std::min(min_dist, dist[i][BOARD_SIZE - 1]);
        }
    }

    return min_dist;
}

double evaluate_position(const Board& board, Colour colour) {
    int my_dist = shortest_path_distance(board, colour);
    int opp_dist = shortest_path_distance(board, opposite_colour(colour));

    // Lower distance is better, so we want opp_dist - my_dist
    return opp_dist - my_dist;
}

} // namespace hex
