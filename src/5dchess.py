import copy
import itertools
import numpy as np

PAWN = 'p'
ROOK = 'r'
KNIGHT = 'n'
BISHOP = 'b'
QUEEN = 'q'
KING = 'k'
EMPTY = ' '
UNMOVED = '*'
PASSANTABLE = '$'

BOARD_SIZE = 8


def piece_id(piece):
    return piece[0].lower()


def as_player(piece, player):
    if player == 0:
        return piece.upper()
    elif player == 1:
        return piece.lower()


def player_of(piece):
    """
    returns player of piece
    """
    if piece is None:
        return None
    elif piece[0].isupper():
        return 0
    elif piece[0].islower():
        return 1
    else:
        return None


def is_unmoved(piece):
    return UNMOVED in piece


def get_moved_piece(piece, idx=(-1, None, -1, -1), end_idx=(-1, None, -1, -1)):
    _, _, i1, j1 = idx
    _, _, i2, j2 = end_idx
    piece = piece[0]  # since it moved
    if piece_id(piece) == PAWN and (i1 > -1):
        if i2 == BOARD_SIZE - 1 or i2 == 0:
            piece = as_player(QUEEN, player_of(piece))
        if abs(i1 - i2) == 2:
            piece = piece + PASSANTABLE
    return piece


def remove_passant(piece):
    return piece.replace(PASSANTABLE, '')


def en_passantable(piece):
    return PASSANTABLE in piece


class Board:
    def __init__(self, pieces=None):
        if pieces is None:
            pieces = [[EMPTY for _ in range(8)] for _ in range(8)]
            for i, row in enumerate((
                    [ROOK + UNMOVED, KNIGHT, BISHOP, QUEEN, KING + UNMOVED, BISHOP, KNIGHT, ROOK + UNMOVED],
                    [PAWN + UNMOVED for _ in range(8)]
            )):
                pieces[i] = [piece.upper() for piece in row]
                pieces[len(pieces) - i - 1] = [piece.lower() for piece in row]
        self.board = pieces

    def get_piece(self, idx):
        i, j = idx
        piece = self.board[i][j]
        return piece

    def add_piece(self, piece, square):
        """
        adds piece to specified square

        :return: (new board with moved piece, captured piece)
        """
        new_board = self.clone()
        i, j = square
        new_board.board[i][j] = piece
        return new_board, self.get_piece(square)

    def remove_piece(self, square):
        """
        removes piece at square from board and returns (new board, piece removed)
        :param square: (i,j)
        :return: (new board, piece removed)
        """
        new_board = self.clone()
        i, j = square
        new_board.board[i][j] = EMPTY
        return new_board, self.get_piece(square)

    def depassant(self, just_moved=None):
        """
        removes the enpassant marker from pieces except for the one indicated by just_moved
        :param just_moved: index
        """
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if (i, j) != just_moved:
                    self.board[i][j] = remove_passant(self.board[i][j])

    def pieces_of(self, player):
        """
        returns iterable of pieces of specified player
        :param player: 0 for white, 1 for black
        :return: iterable of (i,j)
        """
        for (i, row) in enumerate(self.board):
            for (j, piece) in enumerate(row):
                if player == player_of(piece):
                    yield (i, j)

    def clone(self):
        """
        returns copy of self
        """
        return Board(copy.deepcopy(self.board))

    def is_attacked(self, player, idx):
        return False

    def as_array(self):
        pass

    def __str__(self):
        s = ''
        for row in self.board[::-1]:
            s += '|' + ('|'.join([
                get_moved_piece(piece)
                for piece in row])) + '|'
            s += '\n'
            s += '-'*(len(row)*2 + 1)
            s += '\n'
        return s

    @staticmethod
    def empty_string():
        s = ''
        for row in range(16):
            s += ' '*17
            s += '\n'
        return s


class Present:
    def __init__(self, board=None, up_list=None, down_list=None):
        self.board = board
        if up_list is None:
            up_list = []
        if down_list is None:
            down_list = []
        self.up_list = up_list
        self.down_list = down_list

    def get_board(self, dim_idx) -> Board|None:
        if dim_idx > 0:
            dim_idx = dim_idx - 1
            if dim_idx < len(self.up_list):
                return self.up_list[dim_idx]
            else:
                return None
        elif dim_idx < 0:
            dim_idx = -dim_idx - 1
            if dim_idx < len(self.down_list):
                return self.down_list[dim_idx]
            else:
                return None
        else:
            return self.board

    def add_board(self, dim_idx, board):
        if dim_idx > 0:
            dim_idx, listt = dim_idx - 1, self.up_list
        elif dim_idx < 0:
            dim_idx, listt = -dim_idx - 1, self.down_list
        else:
            self.board = board
            return
        while dim_idx >= len(listt):
            listt.append(None)
        listt[dim_idx] = board

    def get_range(self):
        """
        returns range of indices (inclusive)
        """
        return (-len(self.down_list), len(self.up_list))

    def is_empty(self):
        return self.board is None and (self.get_range() == (0, 0))

    def in_range(self, dim):
        bot, top = self.get_range()
        return bot <= dim and dim <= top

    def remove_board(self, dim):
        if self.in_range(dim):
            if dim > 0:
                self.up_list[dim - 1] = None
                while self.up_list and self.up_list[-1] is None:
                    self.up_list.pop()
            elif dim < 0:
                self.down_list[-dim - 1] = None
                while self.down_list and self.down_list[-1] is None:
                    self.down_list.pop()
            else:
                self.board = None

    def __str__(self):
        s = ''
        board_list = self.down_list[::-1] + [self.board] + self.up_list
        str_boards = [Board.empty_string() if board is None else board.__str__()
                      for board in board_list]
        str_boards = [s.split('\n') for s in str_boards]
        for row in range(16):
            s += '\t'.join([str_board[row] for str_board in str_boards])
            s += '\n'
        return s


class Chess5d:
    def __init__(self, present_list=None, first_player=0):
        if present_list is None:
            present_list = [Present(board=Board())]
        self.present_list = present_list
        self.first_player = first_player
        self.overall_range = None
        self._set_overall_range()
        self.move_history = []
        self.dimension_spawn_history = dict()

    def _set_overall_range(self):
        overall_range = [0, 0]
        for present in self.present_list:
            rng = present.get_range()
            overall_range = [min(overall_range[0], rng[0]), max(overall_range[1], rng[1])]
        self.overall_range = overall_range

    def get_active_number(self):
        """
        returns number of potential active timelines in each direction

        i.e. if return is 5, any timeline at most 5 away is active
        """
        return min(self.overall_range[1], -self.overall_range[0]) + 1

    def make_move(self, idx, end_idx):
        """
        moves piece at idx to end idx

        :param idx: (time, dimension, x, y), must be on an existing board
        :param end_idx: (time, dimension, x, y), must be to an existing board, and

        this will edit the game state to remove the piece at idx, move it to end_idx
        """
        if end_idx not in list(self.piece_possible_moves(idx)):
            print("WARNING INVALID MOVE:", idx, '->', end_idx)
        move = (idx, end_idx)
        self.move_history.append(move)

        time1, dim1, i1, j1 = idx
        old_board = self.get_board((time1, dim1))
        new_board, piece = old_board.remove_piece((i1, j1))
        if piece == EMPTY:
            print("WARNING: no piece on square")
        piece = get_moved_piece(piece, idx, end_idx)
        time2, dim2, i2, j2 = end_idx
        if (time1, dim1) == (time2, dim2):
            # here we do not create a new board with the piece removed, as it did no time-space hopping
            new_board, capture = new_board.add_piece(piece, (i2, j2))
            if piece_id(piece) == KING:  # check for castling
                movement = np.max(np.abs(np.array(idx) - np.array(end_idx)))
                if movement > 1:
                    # we have castled, move the rook as well
                    dir = np.sign(j2 - j1)
                    rook_j = 0 if dir == -1 else BOARD_SIZE - 1
                    new_board, rook = new_board.remove_piece((i1, rook_j))
                    new_board, _ = new_board.add_piece(rook, (i1, j1 + dir))
            if piece_id(piece) == PAWN:  # check for en passant
                if abs(i2 - i1) + abs(j2 - j1) == 2:  # captured in xy coords
                    if en_passantable(self.get_piece((time1, dim1, i1, j2))):
                        new_board, capture = new_board.remove_piece((i1, j2))

            new_board.depassant(just_moved=end_idx[2:])
            self.add_board_child((time2, dim2), new_board, move=move)
            return capture
        else:
            # this is the timeline that piece left behind
            new_board.depassant(just_moved=None)  # there are no just moved pawns
            self.add_board_child((time1, dim1), new_board, move=move)
            newer_board, capture = self.get_board((time2, dim2)).add_piece(piece, (i2, j2))
            newer_board.depassant(
                just_moved=None)  # even if this is a pawn, enpassant is not possible with timespace jumps
            self.add_board_child((time2, dim2), newer_board, move=move)
            return capture

    def dimension_made_by(self, td_idx, move):
        return self.dimension_spawn_history[(td_idx, move)]

    def undo_move(self, move=None):
        """
        undos specified move, last move by default
        """
        if move is None:
            if self.move_history:
                move = self.move_history[-1]
            else:
                print('WARNING: no moves to undo')
                return

        def remove_board(td_idx):
            time, dim = td_idx
            self.present_list[time].remove_board(dim)
            if self.present_list[time].is_empty():
                self.present_list.pop(time)

        idx, end_idx = move
        if move not in self.move_history:
            print("ERROR move", idx, '->', end_idx, 'never was made')
            return
        if move != self.move_history[-1]:
            print("WARNING: move", idx, '->', end_idx, 'was not the last move made')
        time1, dim1, i1, j1 = idx
        time2, dim2, i2, j2 = end_idx

        # no matter what, a new board is created when the piece moves from the board at (time1, dim1)
        dim = self.dimension_made_by((time1, dim1), move)
        remove_board((time1 + 1, dim))  # should be right after time1

        if (time1, dim1) != (time2, dim2):
            # if there is a  time dimension jump, another board is created
            dim = self.dimension_made_by((time2, dim2), move)
            remove_board((time2 + 1, dim))  # should be right after time2

        self._set_overall_range()
        self.move_history.remove(move)

    def get_board(self, td_idx):
        time, dim = td_idx
        if time < len(self.present_list):
            board = self.present_list[time].get_board(dim)
            return board

    def get_piece(self, idx):
        (time, dim, i, j) = idx
        board = self.get_board((time, dim))
        if board is not None:
            return board.get_piece((i, j))

    def idx_exists(self, td_idx, ij_idx=(0, 0)) -> bool:
        time, dim = td_idx
        i, j = ij_idx
        if i < 0 or j < 0 or i > 7 or j > 7:
            return False
        if time >= len(self.present_list) or time < 0:
            return False
        else:
            board = self.get_board((time, dim))
            return not (board is None)

    def add_board_child(self, td_idx, board, move):
        """
        adds board as a child to the board specified by td_idx
        :param td_idx: (time, dimension)
        :param board: board to add
        :param move: move that created this board (for undo purposes)
        """
        time, dim = td_idx

        if not self.idx_exists((time + 1, dim)):
            # in this case dimension does not change
            new_dim = dim
        else:
            player = self.player(time=time)
            if player == 0:  # white move
                new_dim = self.overall_range[0] - 1
                self.overall_range[0] -= 1
            else:  # black move
                new_dim = self.overall_range[1] + 1
                self.overall_range[1] += 1

        if len(self.present_list) <= time + 1:
            gift = Present()
            gift.add_board(new_dim, board)
            self.present_list.append(gift)
        else:
            self.present_list[time + 1].add_board(new_dim, board)

        self.dimension_spawn_history[(td_idx, move)] = new_dim

    def board_can_be_moved(self, td_idx):
        """
        returns if the specified board can be moved from
        """
        time, dim = td_idx
        # if the following board exixts, a move has been made, otherwise, no move was made
        return not self.idx_exists((time + 1, dim))

    def boards_with_possible_moves(self):
        """
        iterable of time-dimension coords of boards where a piece can potentially be moved

        equivalent to the leaves of the natural tree
        """
        for t, present in enumerate(self.present_list):
            rng = present.get_range()
            for d in range(rng[0], rng[1] + 1):
                if self.idx_exists((t, d)) and self.board_can_be_moved((t, d)):
                    yield (t, d)

    def pieces_that_can_move(self, player):
        """
        returns iterable of indexes of pieces of player that are able to move

        :return iterable (time, dimension, i, j)
        """
        for (t, d) in self.boards_with_possible_moves():
            if self.player(time=t) == player:
                board = self.get_board((t, d))
                for (i, j) in board.pieces_of(player):
                    yield (t, d, i, j)

    def piece_possible_moves(self, idx):
        """
        returns possible moves of piece at idx
        :param idx: (time, dim, i, j)
        :return: iterable of idx candidates
        """
        idx_time, idx_dim, idx_i, idx_j = idx
        piece = self.get_board((idx_time, idx_dim)).get_piece((idx_i, idx_j))
        pid = piece_id(piece)
        q_k_dims = itertools.chain(*[itertools.combinations(range(4), k) for k in range(1, 5)])

        if pid in (ROOK, BISHOP, QUEEN, KING):  # can move infinitely
            if pid == ROOK:
                dims_to_change = itertools.combinations(range(4), 1)
            elif pid == BISHOP:
                dims_to_change = itertools.combinations(range(4), 2)
            else:
                dims_to_change = q_k_dims
            for dims in dims_to_change:
                for signs in itertools.product((-1, 1), repeat=len(dims)):
                    pos = [idx_time, idx_dim, idx_i, idx_j]
                    vec = np.array((0, 0, 0, 0))
                    for k, dim in enumerate(dims):
                        vec[dim] = signs[k]*((dim == 0) + 1)  # mult by 2 if dim is time
                    pos += vec
                    while self.idx_exists(pos[:2], pos[2:]) and (
                            player_of(piece) != player_of(self.get_piece(pos))):
                        yield tuple(pos)
                        if (player_of(self.get_piece(pos)) is not None) or pid == KING:
                            # end of the line, or the king which moves single spaces
                            break
                        pos += vec
        if pid == KNIGHT:
            dims_to_change = itertools.permutations(range(4), 2)
            for dims in dims_to_change:
                for signs in itertools.product((-1, 1), repeat=len(dims)):
                    pos = [idx_time, idx_dim, idx_i, idx_j]
                    for k, dim in enumerate(dims):
                        # multiply one of the dimensions by 1 and one by 2
                        # can do this with *(k+1)
                        pos[dim] += (k + 1)*signs[k]*((dim == 0) + 1)
                    if self.idx_exists(pos[:2], pos[2:]) and (
                            player_of(piece) != player_of(self.get_piece(pos))):
                        yield tuple(pos)
        if pid == PAWN:
            player = self.player(idx_time)
            dir = -2*player + 1
            # forward moves
            for dim in (2, 1):
                pos = [idx_time, idx_dim, idx_i, idx_j]
                for _ in range(1 + is_unmoved(piece)):
                    pos[dim] += dir
                    if self.idx_exists(pos[:2], pos[2:]) and (player_of(self.get_piece(pos)) is None):
                        yield tuple(pos)
                    else:
                        break
            # diag moves
            for dims in ((2, 3), (1, 0)):
                for aux_sign in (-1, 1):
                    pos = [idx_time, idx_dim, idx_i, idx_j]
                    pos[dims[0]] += dir
                    pos[dims[1]] += aux_sign
                    if self.idx_exists(pos[:2], pos[2:]) and (player_of(self.get_piece(pos)) is not None) and (
                            player_of(piece) != player_of(self.get_piece(pos))):
                        # this MUST be a capture
                        yield tuple(pos)
            # en passant check
            for other_j in (idx_j + 1, idx_j - 1):
                if self.idx_exists((idx_time, idx_dim), (idx_i, other_j)):
                    other_piece = self.get_piece((idx_time, idx_dim, idx_i, other_j))
                    if player_of(other_piece) != player_of(piece) and en_passantable(other_piece):
                        yield (idx_time, idx_dim, idx_i + dir, other_j)
        # castling check
        if pid == KING and is_unmoved(piece):
            for rook_i in (0, BOARD_SIZE - 1):
                for rook_j in (0, BOARD_SIZE - 1):
                    # potential rook squares
                    rook_maybe = self.get_piece((idx_time, idx_dim, rook_i, rook_j))
                    if player_of(rook_maybe) == player_of(piece):
                        if piece_id(rook_maybe) == ROOK and is_unmoved(rook_maybe):
                            dir = np.sign(rook_j - idx_j)
                            works = True
                            for k in range(1, 3):
                                idx_temp = (idx_time, idx_dim, rook_i, idx_j + dir*k)
                                if self.is_attacked(player_of(piece), idx_temp):
                                    works = False
                                if player_of(self.get_piece(idx_temp)) is not None:
                                    works = False
                            if works:
                                yield (idx_time, idx_dim, idx_i, idx_j + 2*dir)

    def all_possible_moves(self, player=None):
        """
        returns an iterable of all possible moves of the specified player
        if player is None, uses the first player that needs to move
        None is included if the player does not NEED to move
        """
        if player is None:
            player = self.player()
        if self.player() != player:
            yield None
        for idx in self.pieces_that_can_move(player=player):
            for end_idx in self.piece_possible_moves(idx):
                yield (idx, end_idx)

    def is_attacked(self, player, idx):
        board = self.get_board(idx[:2])
        board.is_attacked(player, idx[2:])

    def player(self, time=None):
        """
        returns which players turn it is
            if a player does not NEED to move, it is not their turn
        0 for first player
        1 for second player
        :param time: if None, uses present
        """
        if time is None:
            time = self.present()
        # if first player is 0, return time%2
        # if first player is 1, return (time+1)%2
        return (time + self.first_player)%2

    def present(self):
        """
        returns the time index of the present
        """
        active = self.get_active_number()
        return min(t for (t, d) in self.boards_with_possible_moves() if abs(d) <= active)

    def __str__(self):
        s = ''
        for t, present in enumerate(self.present_list):
            str_boards = []
            for dim in range(self.overall_range[0], self.overall_range[1] + 1):
                bored = present.get_board(dim)
                if bored is None:
                    str_boards.append(Board.empty_string())
                else:
                    str_boards.append(bored.__str__())

            str_boards = [s.split('\n') for s in str_boards]

            s += 'time ' + str(t) + ':\n'

            for row in range(16):
                s += '\t'.join([str_board[row] for str_board in str_boards])
                s += '\n'

            s += '\n\n'
        return s


class Chess2d(Chess5d):
    def __init__(self):
        super().__init__()

    def piece_possible_moves(self, idx):
        # only return moves that do not jump time-dimensions
        for end_idx in super().piece_possible_moves(idx):
            if end_idx[:2] == idx[:2]:
                yield end_idx

    def make_move(self, idx, end_idx):
        """
        only need i,j coords
        :param idx: (i,j)
        :param end_idx: (i,j)
        """
        if len(idx) == 4:
            return super().make_move(idx, end_idx)
        else:
            time = len(self.present_list) - 1
            dim = 0
            return super().make_move((time, dim) + tuple(idx), (time, dim) + tuple(end_idx))

    def play(self):
        while True:
            pass


if __name__ == '__main__':
    game = Chess5d()
    print('present', game.present())
    print('capture', game.make_move((0, 0, 1, 3), (0, 0, 3, 3)))
    print()
    print('present', game.present())
    print('capture', game.make_move((1, 0, 6, 4), (1, 0, 4, 4)))
    print()
    print('present', game.present())
    print('capture', game.make_move((2, 0, 0, 1), (0, 0, 2, 1)))
    print()
    print('present', game.present())
    print('capture', game.make_move((1, -1, 6, 6), (1, -1, 5, 6)))
    print()
    print('present', game.present())
    print('capture', game.make_move((2, -1, 1, 7), (2, -1, 2, 7)))
    print()
    print('present', game.present())
    print('capture', game.make_move((3, 0, 6, 6), (3, -1, 6, 6)))
    print()
    print('present', game.present())
    print('capture', game.make_move((4, -1, 2, 1), (4, 0, 4, 1)))
    print()
    print('present', game.present())
    print('capture', game.make_move((5, 0, 4, 4), (5, -1, 4, 4)))
    print()
    print('present', game.present())
    print('capture', game.make_move((6, 0, 4, 1), (6, -1, 4, 3)))
    print()
    print('present', game.present())
    print('capture', game.make_move((7, -1, 7, 1), (7, 0, 5, 1)))
    print()
    print('present', game.present())
    print('capture', game.make_move((8, -1, 0, 1), (8, 0, 2, 1)))
    print()
    print('present', game.present())
    print('capture', game.make_move((9, 0, 5, 1), (9, -1, 7, 1)))
    print()
    print('present', game.present())
    print('capture', game.make_move((10, 0, 2, 1), (10, -1, 0, 1)))
    print()
    print('game:')
    print(game)
    for _ in range(10):
        game.undo_move()
    print('game:')
    print(game)

    quit()
    all_moves = list(game.all_possible_moves(1))
    old_init = None
    for move in all_moves:
        if move is None:
            print(move)
            continue
        idx, end_idx = move
        if idx != old_init:
            print()
            old_init = idx
        print(piece_id(game.get_piece(idx)), idx, '->', end_idx)
