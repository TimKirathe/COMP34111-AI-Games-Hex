#include "transposition_table.h"

namespace hex {

TranspositionTable::TranspositionTable(size_t size)
    : table_(size), size_(size), hits_(0), probes_(0) {
    // Initialize all entries to INVALID state
    // The TTEntry default constructor already sets bound_type to INVALID
    // This ensures a clean initial state
    for (auto& entry : table_) {
        entry.bound_type = BoundType::INVALID;
    }
}

void TranspositionTable::clear() {
    // Mark all entries as invalid without re-allocating memory
    // This is faster than reconstructing the vector
    for (auto& entry : table_) {
        entry.bound_type = BoundType::INVALID;
    }
}

bool TranspositionTable::probe(uint64_t hash, TTEntry& entry) const {
    // Compute table index using modulo
    size_t index = hash % size_;

    // Increment probe counter
    ++probes_;

    // Get entry at computed index
    const TTEntry& stored = table_[index];

    // Check if entry is valid
    if (stored.bound_type == BoundType::INVALID) {
        return false;
    }

    // Verify hash matches (collision detection)
    if (stored.hash != hash) {
        return false;
    }

    // Valid entry found - increment hit counter and copy data
    ++hits_;
    entry = stored;
    return true;
}

void TranspositionTable::store(uint64_t hash, const std::pair<int, int>& best_move,
                                int visits, double wins, int depth, BoundType bound_type) {
    // Compute table index
    size_t index = hash % size_;

    // Get existing entry at this index
    TTEntry& existing = table_[index];

    // Depth-preferred replacement strategy:
    // Replace if:
    // 1. Existing entry is INVALID (empty slot), OR
    // 2. New entry has equal or greater depth (prefer deeper searches)
    if (existing.bound_type == BoundType::INVALID ||
        depth >= existing.depth) {
        // Store all fields
        existing.hash = hash;
        existing.best_move = best_move;
        existing.visits = visits;
        existing.wins = static_cast<float>(wins);  // Explicit cast to float
        existing.depth = static_cast<int16_t>(depth);  // Explicit cast to int16_t
        existing.bound_type = bound_type;
    }
}

double TranspositionTable::hit_rate() const {
    // Avoid division by zero
    if (probes_ == 0) {
        return 0.0;
    }
    return static_cast<double>(hits_) / static_cast<double>(probes_);
}

void TranspositionTable::reset_statistics() {
    hits_ = 0;
    probes_ = 0;
}

} // namespace hex