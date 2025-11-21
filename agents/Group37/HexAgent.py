from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class HexAgent(AgentBase):
    """Group 37's Hex agent.

    TODO: Implement a competitive Hex-playing agent. Consider approaches such as:
    - Minimax with Alpha-Beta pruning
    - Monte Carlo Tree Search (MCTS)
    - Neural network-based evaluation
    - Hybrid approaches combining heuristics with search

    Key methods to implement:
    - _should_swap(): Decide whether to swap on turn 2
    - _select_move(): Choose the best move given the current board state
    - _evaluate_board(): (optional) Evaluate board position for search algorithms
    - _get_valid_moves(): (optional) Generate list of valid moves
    """

    _board_size: int = 11

    def __init__(self, colour: Colour):
        super().__init__(colour)
        # TODO: Initialize any data structures needed for your agent
        # e.g., transposition tables, opening books, neural network models

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move based on the current board state.

        Args:
            turn: The current turn number (starts at 1)
            board: The current board state (READ-ONLY - do not modify!)
            opp_move: The opponent's last move (None if we move first)

        Returns:
            Move: The chosen move - Move(x, y) for placement, Move(-1, -1) for swap
        """
        # Handle swap decision on turn 2
        if turn == 2 and opp_move is not None:
            if self._should_swap(opp_move):
                return Move(-1, -1)

        return self._select_move(board)

    def _should_swap(self, opp_move: Move) -> bool:
        """Decide whether to swap based on opponent's opening move.

        Called only on turn 2 when we're the second player.

        Args:
            opp_move: The opponent's first move

        Returns:
            bool: True to swap colors, False to play normally

        TODO: Implement swap decision logic. Consider:
        - Is the opponent's opening move strong (e.g., near center)?
        - What positions favor swapping vs. playing as second player?
        """
        raise NotImplementedError("TODO: Implement swap decision logic")

    def _select_move(self, board: Board) -> Move:
        """Select the best move given the current board state.

        Args:
            board: The current board state

        Returns:
            Move: The chosen move

        TODO: Implement move selection. Consider:
        - Search algorithms (minimax, MCTS, etc.)
        - Board evaluation heuristics
        - Time management (5 min total limit)
        """
        raise NotImplementedError("TODO: Implement move selection logic")

    # =========================================================================
    # Optional helper methods - implement as needed for your approach
    # =========================================================================

    def _get_empty_tiles(self, board: Board) -> list[tuple[int, int]]:
        """Get all empty tile positions on the board.

        Args:
            board: The current board state

        Returns:
            List of (x, y) tuples for empty positions
        """
        empty = []
        for i in range(self._board_size):
            for j in range(self._board_size):
                if board.tiles[i][j].colour is None:
                    empty.append((i, j))
        return empty

    def _evaluate_board(self, board: Board) -> float:
        """Evaluate the current board position.

        Args:
            board: The current board state

        Returns:
            float: Evaluation score (positive = good for us, negative = bad)

        TODO: Implement board evaluation. Consider:
        - Shortest path to victory for each player
        - Virtual connections and bridge patterns
        - Territory control and influence
        """
        raise NotImplementedError("TODO: Implement board evaluation")

    def _check_win(self, board: Board, colour: Colour) -> bool:
        """Check if the given colour has won.

        Args:
            board: The current board state
            colour: The colour to check for victory

        Returns:
            bool: True if the colour has won

        TODO: Implement win detection using DFS/BFS to find connected path
        - RED wins: connected path from top (row 0) to bottom (row 10)
        - BLUE wins: connected path from left (col 0) to right (col 10)
        """
        raise NotImplementedError("TODO: Implement win detection")