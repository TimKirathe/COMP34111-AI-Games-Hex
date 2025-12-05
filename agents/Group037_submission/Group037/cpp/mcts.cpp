#include "mcts.h"
#include "heuristics.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <limits>

namespace hex {

std::pair<int, int> find_winning_move(const Board& board, Colour colour) {
    auto empty_tiles = board.get_empty_tiles();

    // Only check if few empty tiles (more likely to have winning move)
    if (empty_tiles.size() > 30) {
        return {-1, -1};
    }

    for (const auto& move : empty_tiles) {
        // Make temporary move
        Board temp_board = board;
        temp_board.set_tile(move.first, move.second, colour);

        // Check if we won
        int winner = temp_board.check_winner();
        if ((colour == Colour::RED && winner == 1) ||
            (colour == Colour::BLUE && winner == 2)) {
            return move;
        }
    }

    return {-1, -1};
}

int simulate_with_heuristics(MCTSNode* node, Colour agent_colour) {
    Board board = node->board;
    Colour current_player = node->player_to_move;
    auto empty_tiles = board.get_empty_tiles();

    int move_count = 0;
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    while (!empty_tiles.empty()) {
        // Check for winner periodically
        if (move_count % 5 == 0 || empty_tiles.size() < 10) {
            int winner = board.check_winner();
            if (winner != 0) {
                return winner;
            }
        }

        // Use heuristic-guided selection with 60% probability
        std::pair<int, int> move;
        if (empty_tiles.size() > 5 && dis(gen) < 0.6) {
            move = select_weighted_move(empty_tiles, board, current_player);
        } else {
            // Pure random for speed in late game
            std::uniform_int_distribution<> idx_dist(0, empty_tiles.size() - 1);
            int idx = idx_dist(gen);
            move = empty_tiles[idx];
        }

        // Remove move from empty tiles
        empty_tiles.erase(
            std::remove(empty_tiles.begin(), empty_tiles.end(), move),
            empty_tiles.end()
        );

        // Apply move
        board.set_tile(move.first, move.second, current_player);
        current_player = opposite_colour(current_player);
        move_count++;
    }

    // Final winner check
    return board.check_winner();
}

void backpropagate(MCTSNode* node, int winner, Colour agent_colour) {
    while (node != nullptr) {
        node->visits++;

        if (winner != 0) {
            // The node stores player_to_move, so the PARENT made the move
            // We reward the parent if the winner matches who made the move
            if (node->parent != nullptr) {
                Colour parent_player = node->parent->player_to_move;
                if ((parent_player == Colour::RED && winner == 1) ||
                    (parent_player == Colour::BLUE && winner == 2)) {
                    node->wins += 1.0;
                }
            } else {
                // Root node
                if ((agent_colour == Colour::RED && winner == 1) ||
                    (agent_colour == Colour::BLUE && winner == 2)) {
                    node->wins += 1.0;
                }
            }
        }

        node = node->parent;
    }
}

std::pair<int, int> search(
    const Board& board_state,
    Colour current_player,
    double time_limit
) {
    // Initialize Zobrist hashing
    init_zobrist();

    auto start_time = std::chrono::steady_clock::now();
    double deadline_seconds = time_limit - 0.05; // Reserve 50ms
    auto deadline = start_time + std::chrono::duration<double>(deadline_seconds);

    // Create root node
    MCTSNode root(board_state, current_player);

    // If only one move available, return it immediately
    if (root.untried_moves.size() == 1) {
        return root.untried_moves[0];
    }

    // Check for immediate winning moves
    auto winning_move = find_winning_move(board_state, current_player);
    if (winning_move.first != -1) {
        return winning_move;
    }

    int iterations = 0;
    int best_move_stable_count = 0;
    std::pair<int, int> last_best_move = {-1, -1};

    while (std::chrono::steady_clock::now() < deadline) {
        MCTSNode* node = &root;

        // Selection: traverse tree using UCB1
        while (node->is_fully_expanded() && !node->children.empty()) {
            node = node->best_child();
        }

        // Expansion: add a new child if not terminal
        if (!node->is_terminal() && !node->is_fully_expanded()) {
            node = node->expand();
        }

        // Simulation: heuristic-guided playout
        int winner = simulate_with_heuristics(node, current_player);

        // Backpropagation
        backpropagate(node, winner, current_player);

        iterations++;

        // Early termination check (every 100 iterations)
        if (iterations % 100 == 0 && !root.children.empty()) {
            // Find current best move (most visited)
            auto current_best_it = std::max_element(
                root.children.begin(),
                root.children.end(),
                [](const auto& a, const auto& b) {
                    return a.second->visits < b.second->visits;
                }
            );

            std::pair<int, int> current_best = current_best_it->first;

            if (current_best == last_best_move) {
                best_move_stable_count++;

                // If best move is stable and has high win rate, can terminate early
                if (best_move_stable_count >= 5) {
                    MCTSNode* best_child = current_best_it->second.get();
                    if (best_child->visits > 0) {
                        double win_rate = best_child->wins / best_child->visits;
                        if (win_rate > 0.85 || win_rate < 0.15) {
                            break;
                        }
                    }
                }
            } else {
                best_move_stable_count = 0;
                last_best_move = current_best;
            }
        }
    }

    // Select best move (most visited child)
    if (!root.children.empty()) {
        auto best_it = std::max_element(
            root.children.begin(),
            root.children.end(),
            [](const auto& a, const auto& b) {
                return a.second->visits < b.second->visits;
            }
        );
        return best_it->first;
    }

    // Fallback: use heuristic-guided selection
    auto empty = board_state.get_empty_tiles();
    if (!empty.empty()) {
        auto move = select_weighted_move(empty, board_state, current_player);
        if (move.first != -1) {
            return move;
        }

        // Random fallback
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<> dis(0, empty.size() - 1);
        return empty[dis(gen)];
    }

    return {0, 0}; // Should never happen
}

} // namespace hex
