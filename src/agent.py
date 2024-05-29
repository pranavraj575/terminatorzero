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


def game_outcome(player: Agent,
                 opponent: Agent,
                 game: Chess5d = None,
                 first_player=0,
                 draw_moves=float('inf'),
                 ) -> (int, Chess5d):
    """
    plays through a game and returns outcome (1 if white (player) won, 0 if draw, -1 if black (opponent) won)
    :param game: initial game, if None, initalizes a default game
    :param draw_moves: if this number of moves go by without a capture, returns a draw
    :return: (outcome, resulting game)
    """
    if game is None:
        game = Chess5d()
    player_idx = first_player

    captured = EMPTY
    # while captured == EMPTY:
    bored = 0
    while piece_id(captured) != KING:
        if game.no_moves(player=player_idx):
            break
        current_player = (player, opponent)[player_idx]
        move = current_player.pick_move(game, player_idx)
        if move is END_TURN:
            player_idx = 1 - player_idx
        else:
            captured = game.make_move(move)

        if captured != EMPTY:
            bored = 0
        else:
            bored += 1

        if bored >= draw_moves:
            return 0, game
    # game is now terminal, either player has no moves left, or king was captured
    out = game.clone().terminal_eval()
    return out, game
