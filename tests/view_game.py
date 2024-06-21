from src.chess5d import END_TURN, Chess2d, EMPTY, as_player, KING, QUEEN, Board

if __name__ == '__main__':
    BOARD_SIZE = Board.BOARD_SIZE
    moves = [((0, 0, 0, 2), (0, 0, 1, 3)), 'END_TURN', ((1, 0, 4, 1), (1, 0, 3, 2)), 'END_TURN', ((2, 0, 1, 3), (2, 0
                                                                                                                 , 4,
                                                                                                                 3)),
             'END_TURN', ((3, 0, 3, 2), (3, 0, 3, 1)), 'END_TURN', ((4, 0, 4, 3), (4, 0, 0, 3)), 'END_TURN',
             ((5, 0, 3, 1), (5, 0, 2, 2)), 'END_TURN', ((6, 0, 0, 3), (6, 0, 4, 3)), 'END_TURN', ((7, 0, 2, 2), (7, 0
                                                                                                                 , 3,
                                                                                                                 1)),
             'END_TURN', ((8, 0, 4, 3), (8, 0, 4, 4)), 'END_TURN', ((9, 0, 3, 1), (9, 0, 3, 2)), 'END_TURN',
             ((10, 0, 0, 1), (10, 0, 1, 0)), 'END_TURN', ((11, 0, 3, 2), (11, 0, 2, 3)), 'END_TURN', ((12, 0, 4, 4),
                                                                                                      (12, 0, 4, 2)),
             'END_TURN', ((13, 0, 2, 3), (13, 0, 1, 3)), 'END_TURN', ((14, 0, 1, 0), (14, 0, 2, 1)), 'END_TURN',
             ((15, 0, 1, 3), (15, 0, 2, 3)), 'END_TURN', ((16, 0, 4, 2), (16, 0, 4, 3)), 'END_TURN', ((17,
                                                                                                       0, 2, 3),
                                                                                                      (17, 0, 1, 4)),
             'END_TURN', ((18, 0, 2, 1), (18, 0, 2, 2)), 'END_TURN', ((19, 0, 1, 4), (19, 0, 2, 4)), 'END_TURN',
             ((20, 0, 4, 3), (20, 0, 4, 4)), 'END_TURN', ((21, 0, 2, 4), (21, 0, 1, 3)), 'END_TURN',
             ((22, 0, 2, 2), (22, 0, 1, 3))]

    left = (BOARD_SIZE - 2)//2
    board = Board(pieces=[[EMPTY for _ in range(left)] +
                          [as_player(KING, 0), as_player(QUEEN, 0)] +
                          [EMPTY for _ in range(BOARD_SIZE - 2 - left)]] +
                         [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                         [[EMPTY for _ in range(left)] +
                          [as_player(KING, 1)] +
                          [EMPTY for _ in range(BOARD_SIZE - 1 - left)]]
                  )
    # board = board.flipped_board()
    # board.set_player(1 - board.player)
    # board = board.flipped_board()
    game = Chess2d(board=board, check_validity=True)

    for move in moves:
        game.make_move(move)
    print(game.multiverse)
