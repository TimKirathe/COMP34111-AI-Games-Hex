#ifndef TRANSPOSITION_TABLE_H
#define TRANSPOSITION_TABLE_H

#include <cstdint>
#include <vector>
#include <utility>

namespace hex {

// Bound type for transposition table entries
enum class BoundType : uint8_t {
    EXACT = 0,    // Entry contains exact visit/win data
    INVALID = 1   // Entry is empty or invalid
};

// Transposition table entry - aligned to 64 bytes for cache efficiency
struct alignas(64) TTEntry {
    uint64_t hash;                    // 8 bytes - full Zobrist hash for collision detection
    std::pair<int, int> best_move;    // 8 bytes - (x, y) coordinates of best move
    int visits;                       // 4 bytes - node visit count from MCTS
    float wins;                       // 4 bytes - win count from MCTS
    int16_t depth;                    // 2 bytes - search depth
    BoundType bound_type;             // 1 byte - EXACT or INVALID
    uint8_t padding[35];              // 35 bytes - padding to align struct to 64 bytes

    // Default constructor - initializes to invalid state
    TTEntry() : hash(0), best_move(-1, -1), visits(0), wins(0.0f),
                depth(0), bound_type(BoundType::INVALID) {}
};

// Transposition table for MCTS node caching
// Uses Zobrist hashing for position identification and depth-preferred replacement
class TranspositionTable {
public:
    // Constructor - creates table with specified size
    // Default size: 5,000,000 entries (~320MB memory)
    explicit TranspositionTable(size_t size = 5'000'000);

    // Clear all entries (mark as INVALID without re-allocation)
    void clear();

    // Probe table for a position
    // Returns true if found and valid, false otherwise
    // Entry parameter is populated with data if found
    bool probe(uint64_t hash, TTEntry& entry) const;

    // Store position data in table
    // Uses depth-preferred replacement strategy:
    // - Always replace INVALID entries
    // - Replace valid entries only if new depth >= existing depth
    void store(uint64_t hash, const std::pair<int, int>& best_move,
               int visits, double wins, int depth, BoundType bound_type);

    // Get cache hit rate (hits / total probes)
    double hit_rate() const;

    // Reset hit/probe statistics
    void reset_statistics();

private:
    std::vector<TTEntry> table_;  // Hash table storage
    size_t size_;                 // Table size (constant after construction)
    mutable uint64_t hits_;       // Cache hits (mutable for const probe)
    mutable uint64_t probes_;     // Total probes (mutable for const probe)
};

} // namespace hex

#endif // TRANSPOSITION_TABLE_H