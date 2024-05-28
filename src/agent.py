from src.chess5d import *


class Agent:
    def __init__(self):
        pass

    def pick_move(self, game: Chess5d, player):
        """
        returns a move for player in game
        move is either (idx, end_idx) or None for switching players
        """
        raise NotImplementedError


def outcome(player: Agent, opponent: Agent, game: Chess5d = None, first_player=0):
    """
    plays through a game and returns outcome (1 if player 1 won, 0 if draw, -1 if player 2 won)
    :param game: initial game, if None, initalizes a default game
    :return: (outcome, resulting game)
    """
    if game is None:
        game = Chess5d()
    player_idx = first_player

    stalemate=False
    captured = EMPTY
    #while captured == EMPTY:
    while piece_id(captured)!=KING:
        current_player = (player, opponent)[player_idx]
        move = current_player.pick_move(game, player_idx)
        if move is None:
            player_idx = 1 - player_idx
        else:
            captured = game.make_move(move)
        if game.no_moves(player=player_idx):
            stalemate=True
            break
    if stalemate:
        out=0
    else:
        # the current player won, as they just captured the king
        if player_idx==1:
            out=1
        else:
            out=-1
    return out, game
