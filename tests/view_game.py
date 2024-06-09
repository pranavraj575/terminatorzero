from src.chess5d import END_TURN, Chess2d, EMPTY, as_player, KING, QUEEN, Board, BOARD_SIZE

if __name__ == '__main__':
    moves = [((0, 0, 0, 1), (0, 0, 1, 1)), END_TURN, ((1, 0, 4, 2), (1, 0, 4, 4)), END_TURN, ((2, 0, 1, 1), (2, 0
                                                                                                             , 0, 1)),
             END_TURN, ((3, 0, 4, 4), (3, 0, 2, 2)), END_TURN, ((4, 0, 0, 1), (4, 0, 1, 0)), END_TURN,
             ((5, 0, 2, 2), (5, 0, 4, 2)), END_TURN, ((6, 0, 1, 0), (6, 0, 2, 1)), END_TURN, ((7, 0, 4, 2), (7, 0
                                                                                                             , 4, 4)),
             END_TURN, ((8, 0, 2, 1), (8, 0, 1, 1)), END_TURN, ((9, 0, 4, 4), (9, 0, 1, 1))]

    left = (BOARD_SIZE - 2)//2
    board = Board(pieces=[[EMPTY for _ in range(left)] +
                          [as_player(KING, 0), as_player(QUEEN, 0)] +
                          [EMPTY for _ in range(BOARD_SIZE - 2 - left)]] +
                         [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                         [[EMPTY for _ in range(left)] +
                          [as_player(KING, 1)] +
                          [EMPTY for _ in range(BOARD_SIZE - 1 - left)]]
                  )
    board = board.flipped_board()
    board.set_player(1 - board.player)
    # board = board.flipped_board()
    game = Chess2d(board=board, check_validity=True)

    for move in moves:
        game.make_move(move)
    print(game.multiverse)