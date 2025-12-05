"""C++-accelerated Hex agent using Monte Carlo Tree Search with heuristics."""

import time
import sys
import os
from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move

# Add the current directory to Python path to find libhex_mcts
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    import libhex_mcts
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"WARNING: libhex_mcts C++ module not available: {e}")


class HexAgentCpp(AgentBase):
    """Group 37's C++-accelerated Hex agent using Monte Carlo Tree Search."""

    _board_size: int = 11
    _total_time: float = 300.0  # 5 minutes total
    _time_used: float = 0.0
    _move_count: int = 0

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self._time_used = 0.0
        self._move_count = 0

        if not CPP_AVAILABLE:
            raise RuntimeError(
                "C++ module libhex_mcts not available. "
                "Please compile the C++ extension first."
            )

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move based on the current board state."""
        start_time = time.time()

        # Handle swap decision on turn 2
        if turn == 2 and opp_move is not None:
            if self._should_swap(opp_move):
                return Move(-1, -1)

        move = self._select_move(board, turn)

        # Track time usage
        self._time_used += time.time() - start_time
        self._move_count += 1

        return move

    def _should_swap(self, opp_move: Move) -> bool:
        """Use C++ heuristic-based swap decision."""
        return libhex_mcts.should_swap(opp_move.x, opp_move.y, self._board_size)

    def _get_time_allocation(self, turn: int, board: Board) -> float:
        """
        Dynamically allocate time based on game phase and remaining time.
        """
        remaining_time = self._total_time - self._time_used
        empty_count = sum(
            1 for i in range(board.size) for j in range(board.size)
            if board.tiles[i][j].colour is None
        )
        total_tiles = board.size * board.size

        # Estimate remaining moves (roughly half of empty tiles)
        estimated_moves_left = max(1, empty_count // 2)

        # Base allocation
        base_time = remaining_time / estimated_moves_left

        # Phase-based multiplier
        filled_ratio = 1 - (empty_count / total_tiles)

        if filled_ratio < 0.15:
            # Early game: allocate more time (opening is critical)
            multiplier = 1.5
        elif filled_ratio < 0.4:
            # Mid game: most critical decisions
            multiplier = 1.3
        elif filled_ratio < 0.7:
            # Late-mid game
            multiplier = 1.0
        else:
            # End game: positions are clearer, less time needed
            multiplier = 0.7

        # Emergency time management
        if remaining_time < 30:
            multiplier = 0.3
        elif remaining_time < 60:
            multiplier = 0.5

        allocated = min(base_time * multiplier, remaining_time * 0.15)
        return max(0.1, min(allocated, 10.0))  # Between 0.1s and 10s

    def _select_move(self, board: Board, turn: int) -> Move:
        """Use C++ MCTS to select the best move with time management."""
        time_budget = self._get_time_allocation(turn, board)

        # Convert Python Board to C++ format (flat array)
        board_array = self._board_to_flat_array(board)

        # Convert colour to C++ format (1=RED, 2=BLUE)
        cpp_colour = libhex_mcts.Colour.RED if self._colour == Colour.RED else libhex_mcts.Colour.BLUE

        # Create C++ Board object
        cpp_board = libhex_mcts.Board()
        cpp_board.from_flat_array(board_array)

        # Call C++ MCTS search
        x, y = libhex_mcts.search(cpp_board, cpp_colour, time_budget)

        return Move(x, y)

    def _board_to_flat_array(self, board: Board) -> list[int]:
        """
        Convert Python Board to flat array representation.
        0 = EMPTY, 1 = RED, 2 = BLUE
        """
        arr = []
        for i in range(board.size):
            for j in range(board.size):
                colour = board.tiles[i][j].colour
                if colour is None:
                    arr.append(0)
                elif colour == Colour.RED:
                    arr.append(1)
                else:  # BLUE
                    arr.append(2)
        return arr
