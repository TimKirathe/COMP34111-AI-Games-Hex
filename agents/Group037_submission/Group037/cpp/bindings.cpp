#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mcts.h"
#include "heuristics.h"

namespace py = pybind11;

PYBIND11_MODULE(libhex_mcts, m) {
    m.doc() = "C++ MCTS implementation for Hex game with pybind11 bindings";

    // Expose the main MCTS search function
    m.def("search", &hex::search,
          "Run MCTS and return best move as (x, y)",
          py::arg("board_state"),
          py::arg("current_player"),
          py::arg("time_limit"));

    // Expose swap decision function
    m.def("should_swap", &hex::should_swap,
          "Determine if we should swap based on opponent's opening move",
          py::arg("opp_move_x"),
          py::arg("opp_move_y"),
          py::arg("board_size") = hex::BOARD_SIZE);

    // Expose Board class for Python interaction
    py::class_<hex::Board>(m, "Board")
        .def(py::init<>())
        .def("get_tile", &hex::Board::get_tile)
        .def("set_tile", &hex::Board::set_tile)
        .def("clear_tile", &hex::Board::clear_tile)
        .def("get_empty_tiles", &hex::Board::get_empty_tiles)
        .def("check_winner", &hex::Board::check_winner)
        .def("from_flat_array", &hex::Board::from_flat_array)
        .def("to_flat_array", &hex::Board::to_flat_array)
        .def("size", &hex::Board::size);

    // Expose Colour enum
    py::enum_<hex::Colour>(m, "Colour")
        .value("RED", hex::Colour::RED)
        .value("BLUE", hex::Colour::BLUE)
        .export_values();

    // Expose Tile enum
    py::enum_<hex::Tile>(m, "Tile")
        .value("EMPTY", hex::Tile::EMPTY)
        .value("RED", hex::Tile::RED)
        .value("BLUE", hex::Tile::BLUE)
        .export_values();

    // Initialize Zobrist hashing on module import
    hex::init_zobrist();
}
