import math
import time
from copy import deepcopy
from random import choice

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move
from src.Tile import Tile


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        board: Board,
        player_to_move: Colour,
        parent: "MCTSNode | None" = None,
        move: tuple[int, int] | None = None,
    ):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move = move  # Move that led to this node
        self.children: dict[tuple[int, int], "MCTSNode"] = {}
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = self._get_empty_tiles()

    def _get_empty_tiles(self) -> list[tuple[int, int]]:
        """Get all empty positions on the board."""
        empty = []
        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.tiles[i][j].colour is None:
                    empty.append((i, j))
        return empty

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state (someone won or board full)."""
        return self._check_winner() is not None or len(self._get_empty_tiles()) == 0

    def _check_winner(self) -> Colour | None:
        """Check if there's a winner. Returns the winning color or None."""
        # Check RED (top to bottom)
        if self._has_won(Colour.RED):
            return Colour.RED
        # Check BLUE (left to right)
        if self._has_won(Colour.BLUE):
            return Colour.BLUE
        return None

    def _has_won(self, colour: Colour) -> bool:
        """Check if the given color has won using DFS."""
        size = self.board.size
        visited = set()

        if colour == Colour.RED:
            # Start from top row, try to reach bottom row
            start_positions = [
                (0, j)
                for j in range(size)
                if self.board.tiles[0][j].colour == colour
            ]
            target_row = size - 1

            def is_goal(x, y):
                return x == target_row

        else:  # BLUE
            # Start from left column, try to reach right column
            start_positions = [
                (i, 0)
                for i in range(size)
                if self.board.tiles[i][0].colour == colour
            ]
            target_col = size - 1

            def is_goal(x, y):
                return y == target_col

        # DFS from each starting position
        for start in start_positions:
            if start in visited:
                continue
            stack = [start]
            while stack:
                x, y = stack.pop()
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                if is_goal(x, y):
                    return True
                # Check neighbors
                for k in range(Tile.NEIGHBOUR_COUNT):
                    nx = x + Tile.I_DISPLACEMENTS[k]
                    ny = y + Tile.J_DISPLACEMENTS[k]
                    if 0 <= nx < size and 0 <= ny < size:
                        if (nx, ny) not in visited:
                            if self.board.tiles[nx][ny].colour == colour:
                                stack.append((nx, ny))
        return False

    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> "MCTSNode":
        """Select the best child according to UCB1."""
        return max(
            self.children.values(),
            key=lambda c: c.ucb1(exploration_constant),
        )

    def expand(self) -> "MCTSNode":
        """Expand by adding a new child node for an untried move."""
        move = self.untried_moves.pop()
        new_board = deepcopy(self.board)
        new_board.set_tile_colour(move[0], move[1], self.player_to_move)
        next_player = Colour.opposite(self.player_to_move)
        child = MCTSNode(new_board, next_player, parent=self, move=move)
        self.children[move] = child
        return child


class HexAgent(AgentBase):
    """Group 37's Hex agent using Monte Carlo Tree Search."""

    _board_size: int = 11
    _time_per_move: float = 2.5  # seconds per move

    def __init__(self, colour: Colour):
        super().__init__(colour)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move based on the current board state."""
        # Handle swap decision on turn 2
        if turn == 2 and opp_move is not None:
            if self._should_swap(opp_move):
                return Move(-1, -1)

        return self._select_move(board)

    def _should_swap(self, opp_move: Move) -> bool:
        """Swap if opponent's move is within 2 tiles of center."""
        center = self._board_size // 2
        dist = abs(opp_move.x - center) + abs(opp_move.y - center)
        return dist <= 2

    def _select_move(self, board: Board) -> Move:
        """Use MCTS to select the best move."""
        start_time = time.time()
        deadline = start_time + self._time_per_move - 0.1  # Reserve 100ms

        # Create root node with current player
        root = MCTSNode(deepcopy(board), self._colour)

        # If only one move available, return it immediately
        if len(root.untried_moves) == 1:
            move = root.untried_moves[0]
            return Move(move[0], move[1])

        iterations = 0
        while time.time() < deadline:
            node = root

            # Selection: traverse tree using UCB1
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion: add a new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: random playout
            winner = self._simulate(node)

            # Backpropagation
            self._backpropagate(node, winner)

            iterations += 1

        # Select best move (most visited child)
        if root.children:
            best_move = max(root.children.keys(), key=lambda m: root.children[m].visits)
            return Move(best_move[0], best_move[1])

        # Fallback: random empty tile
        empty = self._get_empty_tiles(board)
        if empty:
            move = choice(empty)
            return Move(move[0], move[1])
        return Move(0, 0)  # Should never happen

    def _simulate(self, node: MCTSNode) -> Colour | None:
        """Simulate a random game from the given node."""
        board = deepcopy(node.board)
        current_player = node.player_to_move
        empty_tiles = [
            (i, j)
            for i in range(board.size)
            for j in range(board.size)
            if board.tiles[i][j].colour is None
        ]

        # Random playout
        while empty_tiles:
            # Check for winner periodically (every 5 moves for speed)
            if len(empty_tiles) % 5 == 0 or len(empty_tiles) < 10:
                winner = self._check_winner_fast(board)
                if winner is not None:
                    return winner

            idx = choice(range(len(empty_tiles)))
            move = empty_tiles.pop(idx)
            board.set_tile_colour(move[0], move[1], current_player)
            current_player = Colour.opposite(current_player)

        # Final winner check
        return self._check_winner_fast(board)

    def _check_winner_fast(self, board: Board) -> Colour | None:
        """Fast winner check using DFS."""
        size = board.size
        # Check RED
        visited = set()
        for j in range(size):
            if board.tiles[0][j].colour == Colour.RED and (0, j) not in visited:
                if self._dfs_win(board, 0, j, Colour.RED, visited, size):
                    return Colour.RED
        # Check BLUE
        visited = set()
        for i in range(size):
            if board.tiles[i][0].colour == Colour.BLUE and (i, 0) not in visited:
                if self._dfs_win(board, i, 0, Colour.BLUE, visited, size):
                    return Colour.BLUE
        return None

    def _dfs_win(
        self,
        board: Board,
        x: int,
        y: int,
        colour: Colour,
        visited: set,
        size: int,
    ) -> bool:
        """DFS to check if color has won."""
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            # Goal check
            if colour == Colour.RED and cx == size - 1:
                return True
            if colour == Colour.BLUE and cy == size - 1:
                return True
            # Neighbors
            for k in range(Tile.NEIGHBOUR_COUNT):
                nx = cx + Tile.I_DISPLACEMENTS[k]
                ny = cy + Tile.J_DISPLACEMENTS[k]
                if 0 <= nx < size and 0 <= ny < size:
                    if (nx, ny) not in visited and board.tiles[nx][ny].colour == colour:
                        stack.append((nx, ny))
        return False

    def _backpropagate(self, node: MCTSNode, winner: Colour | None) -> None:
        """Backpropagate the result up the tree."""
        while node is not None:
            node.visits += 1
            if winner is not None:
                # The node stores player_to_move, so the PARENT made the move
                # We reward the parent if the winner matches who made the move
                if node.parent is not None:
                    parent_player = node.parent.player_to_move
                    if winner == parent_player:
                        node.wins += 1
                else:
                    # Root node
                    if winner == self._colour:
                        node.wins += 1
            node = node.parent

    def _get_empty_tiles(self, board: Board) -> list[tuple[int, int]]:
        """Get all empty tile positions on the board."""
        empty = []
        for i in range(self._board_size):
            for j in range(self._board_size):
                if board.tiles[i][j].colour is None:
                    empty.append((i, j))
        return empty

    def _check_win(self, board: Board, colour: Colour) -> bool:
        """Check if the given colour has won."""
        visited = set()
        size = board.size
        if colour == Colour.RED:
            for j in range(size):
                if board.tiles[0][j].colour == colour and (0, j) not in visited:
                    if self._dfs_win(board, 0, j, colour, visited, size):
                        return True
        else:
            for i in range(size):
                if board.tiles[i][0].colour == colour and (i, 0) not in visited:
                    if self._dfs_win(board, i, 0, colour, visited, size):
                        return True
        return False