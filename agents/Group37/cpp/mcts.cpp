#include "mcts.h"
#include "heuristics.h"
#include "transposition_table.h"
#include <chrono>
#include <random>
#include <algorithm>
#include <limits>
#include <iostream>

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

void backpropagate(MCTSNode* node, int winner, Colour agent_colour,
                   TranspositionTable& tt) {
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

        // Store updated statistics in transposition table
        if (node->get_hash() != 0) {
            // Compute depth (distance from leaf)
            int depth = 0;
            MCTSNode* temp = node;
            while (temp->parent != nullptr) {
                depth++;
                temp = temp->parent;
            }

            // Find best child move if node has children (for best_move storage)
            std::pair<int, int> best_move = {-1, -1};
            if (!node->children.empty()) {
                auto best_it = std::max_element(
                    node->children.begin(),
                    node->children.end(),
                    [](const auto& a, const auto& b) {
                        return a.second->visits < b.second->visits;
                    }
                );
                best_move = best_it->first;
            }

            // Store in transposition table
            tt.store(node->get_hash(), best_move, node->visits,
                     node->wins, depth, BoundType::EXACT);
        }

        node = node->parent;
    }
}

/**
 * MCTS search with transposition table integration.
 *
 * Transposition Table Features:
 * - Zobrist hashing for position identification
 * - Depth-preferred replacement strategy
 * - Statistical blending during selection (blend_factor=0.7, min_visits=10)
 * - Move ordering during expansion (min_visits=5)
 * - Automatic storage during backpropagation
 *
 * Memory: ~320MB for 5 million entries
 * Expected improvement: 5-40% faster convergence depending on game phase
 */
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

    // Create transposition table for position caching
    TranspositionTable tt(5'000'000);

    // Compute Zobrist hash for root position
    uint64_t root_hash = compute_board_hash(board_state);
    MCTSNode root(board_state, current_player, nullptr, {-1, -1}, root_hash);

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

        // Check transposition table for this position
        TTEntry cached_entry;
        if (node->get_hash() != 0 && tt.probe(node->get_hash(), cached_entry)) {
            // Only use cached data if it's from sufficient search depth
            if (cached_entry.bound_type == BoundType::EXACT &&
                cached_entry.visits >= 10) {  // Minimum visit threshold

                // Update node statistics with cached values
                // We blend rather than replace to maintain tree consistency
                double blend_factor = 0.7;  // Weight towards cached data
                node->visits = static_cast<int>(
                    node->visits * (1 - blend_factor) +
                    cached_entry.visits * blend_factor
                );
                node->wins = node->wins * (1 - blend_factor) +
                             cached_entry.wins * blend_factor;
            }
        }

        // Expansion: add a new child if not terminal
        if (!node->is_terminal() && !node->is_fully_expanded()) {
            std::pair<int, int> tt_best_move = {-1, -1};
            TTEntry cached_entry;
            if (node->get_hash() != 0 && tt.probe(node->get_hash(), cached_entry)) {
                if (cached_entry.bound_type == BoundType::EXACT &&
                    cached_entry.visits >= 5) {
                    tt_best_move = cached_entry.best_move;
                }
            }
            node = node->expand(tt_best_move);
        }

        // Simulation: heuristic-guided playout
        int winner = simulate_with_heuristics(node, current_player);

        // Backpropagation
        backpropagate(node, winner, current_player, tt);

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

#ifdef TT_DEBUG
    // Transposition table statistics for debugging
    std::cerr << "=== Transposition Table Statistics ===" << std::endl;
    std::cerr << "  Hit Rate: " << (tt.hit_rate() * 100.0) << "%" << std::endl;
    std::cerr << "======================================" << std::endl;
#endif

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
