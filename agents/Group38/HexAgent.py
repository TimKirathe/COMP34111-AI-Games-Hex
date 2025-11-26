import copy
import math
from random import choice, random
from time import perf_counter_ns as time

from src.AgentBase import AgentBase
from src.Board import Board
from src.Colour import Colour
from src.Move import Move


class MCTSNode:
    """Node in the MCTS tree with RAVE support."""

    def __init__(self, board: Board, move: Move | None, parent: "MCTSNode | None", colour: Colour):
        self.board = board
        self.move = move
        self.parent = parent
        self.colour = colour
        self.children = []
        self.wins = 0
        self.visits = 0
        # RAVE statistics
        self.rave_wins = 0
        self.rave_visits = 0
        self.untried_moves = None  # Lazy initialization

    def get_untried_moves(self, board: Board, heuristic_sort: bool = True) -> list[Move]:
        """Get valid moves, optionally sorted by heuristic."""
        if self.untried_moves is None:
            moves = []
            center = board.size // 2
            for i in range(board.size):
                for j in range(board.size):
                    if board.tiles[i][j].colour is None:
                        if heuristic_sort:
                            # Prioritize center and nearby tiles
                            dist = abs(i - center) + abs(j - center)
                            moves.append((dist, Move(i, j)))
                        else:
                            moves.append(Move(i, j))
            
            if heuristic_sort:
                moves.sort(key=lambda x: x[0])
                self.untried_moves = [m[1] for m in moves]
            else:
                self.untried_moves = moves
        
        return self.untried_moves

    def is_fully_expanded(self, board: Board) -> bool:
        """Check if all child nodes have been expanded."""
        moves = self.get_untried_moves(board)
        return len(moves) == 0

    def best_child(self, exploration_weight: float = 1.41, rave_weight: float = 0.5) -> "MCTSNode":
        """Select best child using UCB1 + RAVE formula."""
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            # Standard UCB1
            ucb_score = child.wins / child.visits
            ucb_score += exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            
            # RAVE component
            if child.rave_visits > 0:
                rave_score = child.rave_wins / child.rave_visits
                # Blend UCB and RAVE based on visit count
                beta = child.rave_visits / (child.visits + child.rave_visits + 1e-5)
                score = (1 - beta) * ucb_score + beta * rave_score
            else:
                score = ucb_score
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child

    def expand(self, board: Board) -> "MCTSNode":
        """Expand a high-priority untried move."""
        moves = self.get_untried_moves(board)
        if not moves:
            return self
        
        move = moves.pop(0)  # Take highest priority move
        new_board = copy.deepcopy(board)
        new_board.set_tile_colour(move.x, move.y, self.colour)
        child_node = MCTSNode(new_board, move, self, Colour.opposite(self.colour))
        self.children.append(child_node)
        return child_node


class HexAgent(AgentBase):
    """Group 37's enhanced MCTS Hex agent with RAVE and heuristics."""

    _board_size: int = 11
    
    # Opening book: strong first moves
    OPENING_BOOK = {
        1: [(5, 5), (5, 6), (6, 5)],  # Center positions for first move
    }

    def __init__(self, colour: Colour):
        super().__init__(colour)
        self.total_time_used = 0
        self.max_time = 5 * 60 * 10**9  # 5 minutes in nanoseconds (tournament limit)

    def make_move(self, turn: int, board: Board, opp_move: Move | None) -> Move:
        """Make a move based on the current board state."""
        start = time()
        
        # Use opening book for first move
        if turn == 1:
            move = Move(*choice(self.OPENING_BOOK[1]))
            self.total_time_used += time() - start
            return move
        
        # Handle swap decision on turn 2
        if turn == 2 and opp_move is not None:
            if self._should_swap(opp_move):
                self.total_time_used += time() - start
                return Move(-1, -1)

        move = self._select_move(board, turn)
        self.total_time_used += time() - start
        return move

    def _should_swap(self, opp_move: Move) -> bool:
        """Decide whether to swap based on opponent's opening move."""
        center = self._board_size // 2
        # Calculate distance from center (Manhattan distance)
        dist = abs(opp_move.x - center) + abs(opp_move.y - center)
        # Swap if within 3 tiles of center (strong positions)
        return dist <= 3

    def _select_move(self, board: Board, turn: int) -> Move:
        """Select the best move using enhanced MCTS with RAVE."""
        # Dynamic time allocation - optimized for speed scoring (25% of grade)
        remaining_time = self.max_time - self.total_time_used
        empty_tiles = len(self._get_empty_tiles(board))
        
        # More aggressive time allocation for faster moves
        if empty_tiles > 90:  # Very early game - play fast
            time_budget = min(remaining_time * 0.02, 2 * 10**9)
        elif empty_tiles > 70:  # Early game
            time_budget = min(remaining_time * 0.04, 4 * 10**9)
        elif empty_tiles > 40:  # Mid game
            time_budget = min(remaining_time * 0.08, 8 * 10**9)
        else:  # End game - more critical
            time_budget = min(remaining_time * 0.15, 15 * 10**9)
        
        time_budget = max(time_budget, 0.5 * 10**9)  # Minimum 0.5 seconds for speed

        # Run MCTS with RAVE
        root = MCTSNode(board, None, None, self.colour)
        start_time = time()
        iterations = 0

        while (time() - start_time) < time_budget:
            node = self._select(root, board)
            
            if not self._is_game_over(node.board):
                if node.visits > 0 and not node.is_fully_expanded(node.board):
                    node = node.expand(node.board)
                
                winner, moves_played = self._simulate(node.board, node.colour)
            else:
                winner = self._get_winner(node.board)
                moves_played = []
            
            self._backpropagate(node, winner, moves_played)
            iterations += 1

        # Select best move
        if not root.children:
            empty = self._get_empty_tiles(board)
            return Move(empty[0][0], empty[0][1]) if empty else Move(0, 0)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _select(self, node: MCTSNode, board: Board) -> MCTSNode:
        """Selection phase: traverse tree using UCB1 + RAVE."""
        current_board = board
        while not self._is_game_over(node.board):
            if not node.is_fully_expanded(node.board):
                return node
            else:
                node = node.best_child()
                current_board = node.board
        return node

    def _simulate(self, board: Board, current_colour: Colour) -> tuple[Colour | None, list[Move]]:
        """Simulation with heuristic-guided playouts."""
        sim_board = copy.deepcopy(board)
        sim_colour = current_colour
        moves_played = []
        max_moves = 200  # Prevent infinite loops

        for _ in range(max_moves):
            if self._is_game_over(sim_board):
                break
            
            empty = self._get_empty_tiles(sim_board)
            if not empty:
                break
            
            # Heuristic-guided move selection (80% heuristic, 20% random)
            if random() < 0.8:
                move = self._heuristic_move(sim_board, empty, sim_colour)
            else:
                move = choice(empty)
            
            sim_board.set_tile_colour(move[0], move[1], sim_colour)
            moves_played.append(Move(move[0], move[1]))
            
            if self._check_win(sim_board, sim_colour):
                return sim_colour, moves_played
            
            sim_colour = Colour.opposite(sim_colour)

        return self._get_winner(sim_board), moves_played

    def _heuristic_move(self, board: Board, empty_tiles: list[tuple[int, int]], colour: Colour) -> tuple[int, int]:
        """Select move using simple heuristics."""
        center = self._board_size // 2
        best_score = -float('inf')
        best_move = empty_tiles[0]
        
        for x, y in empty_tiles:
            score = 0
            
            # Prefer center
            dist_from_center = abs(x - center) + abs(y - center)
            score -= dist_from_center * 0.5
            
            # Prefer connecting to own pieces
            neighbors = [(0, 1), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1)]
            own_neighbors = 0
            opp_neighbors = 0
            
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self._board_size and 0 <= ny < self._board_size:
                    tile_colour = board.tiles[nx][ny].colour
                    if tile_colour == colour:
                        own_neighbors += 1
                    elif tile_colour is not None:
                        opp_neighbors += 1
            
            score += own_neighbors * 3
            score += opp_neighbors * 1.5  # Blocking is also valuable
            
            # Prefer progress toward goal
            if colour == Colour.RED:
                score += (self._board_size - x) * 0.3  # Progress toward bottom
            else:
                score += (self._board_size - y) * 0.3  # Progress toward right
            
            if score > best_score:
                best_score = score
                best_move = (x, y)
        
        return best_move

    def _backpropagate(self, node: MCTSNode, winner: Colour | None, moves_played: list[Move]):
        """Backpropagation with RAVE updates."""
        # Standard backpropagation
        current = node
        while current is not None:
            current.visits += 1
            if winner == self.colour:
                current.wins += 1
            elif winner == Colour.opposite(self.colour):
                current.wins -= 1
            
            # RAVE updates: update all nodes that could have played these moves
            for move in moves_played:
                for child in current.children:
                    if child.move and child.move.x == move.x and child.move.y == move.y:
                        child.rave_visits += 1
                        if winner == self.colour:
                            child.rave_wins += 1
                        elif winner == Colour.opposite(self.colour):
                            child.rave_wins -= 1
            
            current = current.parent

    def _get_empty_tiles(self, board: Board) -> list[tuple[int, int]]:
        """Get all empty tile positions on the board."""
        empty = []
        for i in range(self._board_size):
            for j in range(self._board_size):
                if board.tiles[i][j].colour is None:
                    empty.append((i, j))
        return empty

    def _is_game_over(self, board: Board) -> bool:
        """Check if the game is over."""
        return self._check_win(board, Colour.RED) or self._check_win(board, Colour.BLUE)

    def _get_winner(self, board: Board) -> Colour | None:
        """Get the winner of the game."""
        if self._check_win(board, Colour.RED):
            return Colour.RED
        if self._check_win(board, Colour.BLUE):
            return Colour.BLUE
        return None

    def _check_win(self, board: Board, colour: Colour) -> bool:
        """Check if the given colour has won using optimized DFS."""
        # Use board's built-in win detection (it's already optimized)
        return board.has_ended(colour)