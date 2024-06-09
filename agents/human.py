from src.agent import Agent
from src.chess5d import Chess5d, END_TURN, piece_id


class Human(Agent):
    """
    takes input for move
    """

    def __init__(self):
        super().__init__()

    def pick_move(self, game: Chess5d, player):
        print(game.__str__())
        moves = list(game.all_possible_moves(player=player))

        if len(moves) == 1:
            return moves[0]
        if END_TURN in moves:
            end = input('end turn? [y/n]: ')
            if end.lower() == 'y':
                return END_TURN
            moves.remove(END_TURN)

        starters = list(set(idx for idx, _ in moves))

        for i, starter in enumerate(starters):
            print(i, ':', piece_id(game.get_piece(starter)), starter[2:], ' board ', starter[:2])
        choice = ''
        while not choice.isnumeric() or int(choice) < 0 or int(choice) >= len(starters):
            choice = input('type which piece to move: ')
        starter = starters[int(choice)]
        finishers = list(set(end_idx for idx, end_idx in moves if idx == starter))

        for i, finisher in enumerate(finishers):
            print(i, ':', finisher[2:], ' board ', finisher[:2])
        choice = ''
        while not choice.isnumeric() or int(choice) < 0 or int(choice) >= len(finishers):
            choice = input('type where to move it: ')
        finisher = finishers[int(choice)]
        test_game = game.clone()
        move = (starter, finisher)
        test_game.make_move(move)
        if input('display result? [y/n]: ').lower() == 'y':
            print(test_game)
            if input('redo move? [y/n]: ').lower() == 'y':
                return self.pick_move(game=game, player=player)
        return move


if __name__ == '__main__':
    from src.agent import game_outcome
    from agents.non_learning import Randy

    game = Chess5d(check_validity=True)

    outcome, _ = game_outcome(Human(), Randy(), game=game)
    if outcome == 0:
        print('draw')
    elif outcome == 1:
        print('you won')
    else:
        print('you lost')
