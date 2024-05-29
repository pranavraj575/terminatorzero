from src.agent import Agent
from src.chess5d import Chess5d
import random


class Randy(Agent):
    """
    random move
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, game: Chess5d, player):
        all_moves = list(game.all_possible_moves(player=player))
        move = all_moves[random.randint(0, len(all_moves) - 1)]
        return move


class Randathan(Agent):
    """
    random move, but only on boards required to play
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, game: Chess5d, player):
        all_moves = list(game.all_possible_moves(player=player))
        if None in all_moves:
            return None
        moves_on_active_boards = [(idx, end_idx) for (idx, end_idx) in all_moves
                                  if game.dim_is_active(dim=idx[1]) or game.dim_is_active(dim=end_idx[1])]
        if moves_on_active_boards:
            move = moves_on_active_boards[random.randint(0, len(moves_on_active_boards) - 1)]
        else:
            move = all_moves[random.randint(0, len(all_moves) - 1)]
        return move


class FastGuy(Agent):
    """
    first move
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, game: Chess5d, player):
        return next(game.all_possible_moves(player=player))


if __name__ == '__main__':
    from src.agent import game_outcome
    from src.utilitites import seed_all
    from src.chess5d import Chess2d

    seed_all(1)
    winner, game = game_outcome(FastGuy(), FastGuy(), Chess2d(), draw_moves=100)
    print(game.multiverse)
    for move in (game.move_history):
        print(move)
    print(winner)
