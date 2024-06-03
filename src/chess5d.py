import copy
import itertools
import numpy as np

PAWN = 'p'
ROOK = 'r'
KNIGHT = 'n'
BISHOP = 'b'
QUEEN = 'q'  # attacks on any number of diagonals
KING = 'k'

PRINCESS = 'c'  # combo of rook and bishop
UNICORN = 'u'  # attacks on triagonls
DRAGON = 'd'  # attacks on quadragonals
BRAWN = 'b'  # beefy pawn, unimplemented (the en brasant and capturing is annoying)

USING_PIECES = [ROOK, KNIGHT, BISHOP, QUEEN, KING, PAWN]

PROMOTION = QUEEN

EMPTY = ' '
UNMOVED = '*'
PASSANTABLE = '$'

NUM_PIECES = len(USING_PIECES)
BOARD_SIZE = 8

END_TURN = 'END_TURN'
PASS_TURN = 'PASS_TURN'


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
            piece = as_player(PROMOTION, player_of(piece))
        if abs(i1 - i2) == 2:
            piece = piece + PASSANTABLE
    return piece


def remove_passant(piece):
    return piece.replace(PASSANTABLE, '')


def en_passantable(piece):
    return PASSANTABLE in piece


class Board:
    def __init__(self, pieces=None, player=0):
        self.player = player
        if pieces is None:
            pieces = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            if BOARD_SIZE == 8:
                back_rank = [ROOK + UNMOVED, KNIGHT, BISHOP, QUEEN, KING + UNMOVED, BISHOP, KNIGHT, ROOK + UNMOVED]
            elif BOARD_SIZE == 5:
                back_rank = [ROOK + UNMOVED, KNIGHT, BISHOP, QUEEN, KING + UNMOVED]
            else:
                raise Exception("NO DEFAULT FOR BOARD SIZE " + str(BOARD_SIZE))
            for i, row in enumerate((
                    back_rank,
                    [PAWN + UNMOVED for _ in range(BOARD_SIZE)]
            )):
                pieces[i] = [piece.upper() for piece in row]
                pieces[len(pieces) - i - 1] = [piece.lower() for piece in row]
        self.board = pieces

    def flipped_board(self):
        return Board(pieces=[[as_player(piece, 1 - player_of(piece)) if piece != EMPTY else EMPTY
                              for piece in row]
                             for row in self.board[::-1]],
                     player=1 - self.player)

    def set_player(self, player):
        self.player = player

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

    def all_pieces(self):
        """
        returns iterable of all pieces locations on the board
        :return: iterable of (i,j)
        """
        for (i, row) in enumerate(self.board):
            for (j, piece) in enumerate(row):
                if player_of(piece) is not None:
                    yield (i, j)

    def pieces_of(self, player):
        """
        returns iterable of pieces of specified player
        :param player: 0 for white, 1 for black
        :return: iterable of (i,j)
        """
        for idx in self.all_pieces():
            piece = self.get_piece(idx)
            if player == player_of(piece):
                yield idx

    def clone(self):
        """
        returns copy of self
        """
        return Board(pieces=copy.deepcopy(self.board), player=self.player)

    @staticmethod
    def encoding_shape():
        return (BOARD_SIZE, BOARD_SIZE, 2*NUM_PIECES + 4)

    @staticmethod
    def blocked_board_encoding():
        encoded = np.zeros(Board.encoding_shape())
        return encoded

    @staticmethod
    def is_blocked(encoding):
        return encoding[0, 0, 2*NUM_PIECES + 3] == 0

    def encoding(self):
        encoded = np.zeros(Board.encoding_shape())
        encoder_dict = {piece: i for (i, piece) in enumerate(USING_PIECES)}

        def piece_dimension(pid, player):
            return encoder_dict[pid] + NUM_PIECES*player

        kings_unmoved = [False, False]
        left_rooks_unmoved = [False, False]
        right_rooks_unmoved = [False, False]

        k_0 = 2*NUM_PIECES

        encoded[:, :, k_0] = self.player
        encoded[:, :, 2*NUM_PIECES + 3] = 1  # unblocked board marker
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] != EMPTY:
                    piece = self.get_piece((i, j))
                    pid = piece_id(piece)
                    encoded[i, j, piece_dimension(pid, player_of(piece))] = 1

                    encoded[i, j, k_0 + 1] = is_unmoved(piece)
                    encoded[i, j, k_0 + 2] = en_passantable(piece)
        return encoded

    @staticmethod
    def decoding(encoding):
        piece_order = USING_PIECES

        if Board.is_blocked(encoding):
            return None

        pieces = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        k_0 = 2*NUM_PIECES
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = EMPTY
                for k in range(k_0):
                    if int(encoding[i, j, k]) == 1:
                        player = k//NUM_PIECES
                        pid = piece_id(piece_order[k%NUM_PIECES])
                        piece = as_player(pid, player)
                        if encoding[i, j, k_0 + 1]:
                            piece += UNMOVED
                        if encoding[i, j, k_0 + 2]:
                            piece += PASSANTABLE
                pieces[i][j] = piece
        return Board(pieces=pieces, player=int(encoding[0, 0, k_0]))

    def compressed(self):
        return (self.player, '\n'.join(['\t'.join(row) for row in self.board]))

    @staticmethod
    def decompress(compression):
        player, board_compression = compression
        board = [row.split('\t') for row in board_compression.split('\n')]
        return Board(pieces=board, player=player)

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
        for row in range(2*BOARD_SIZE):
            s += ' '*(2*BOARD_SIZE + 1)
            s += '\n'
        return s


class Timeline:
    def __init__(self, board_list: [Board] = None, start_idx: int = 0):
        if board_list is None:
            board_list = []
        self.start_idx = start_idx
        self.board_list = board_list

    def is_empty(self):
        return len(self.board_list) == 0

    def clone(self):
        return Timeline(
            board_list=[board.clone() for board in self.board_list],
            start_idx=self.start_idx,
        )

    def flip_timeline(self):
        self.board_list = [board.flipped_board() for board in self.board_list]

    def get_board(self, real_time: int) -> Board | None:
        """
        gets board at time, where time is the 'real' index, relative to 0
        """
        if self.in_range(real_time=real_time):
            return self.board_list[real_time - self.start_idx]
        else:
            return None

    def get_time_range(self):
        """
        valid time range [start, end)
        """
        return (self.start_idx, self.start_idx + len(self.board_list))

    def end_time(self):
        """
        last possible time
        """
        return self.start_idx + len(self.board_list) - 1

    def in_range(self, real_time):
        start, end = self.get_time_range()
        return start <= real_time and real_time < end

    def append(self, board: Board, time_idx=None):
        # if not self.in_range(time_idx):
        #    raise Exception("INVALID INDEX")
        self.board_list.append(board)

    def pop(self):
        self.board_list.pop()

    def compressed(self):
        return (
            self.start_idx,
            tuple(board.compressed() for board in self.board_list),
        )

    @staticmethod
    def decompress(compression):
        start_idx, compressed_board_list = compression
        board_list = [Board.decompress(compressed_board) for compressed_board in compressed_board_list]
        return Timeline(board_list=board_list, start_idx=start_idx)

    def __str__(self):
        s = ''
        interspace = len(str(self.end_time())) + 6

        str_boards = [Board.empty_string()
                      for _ in range(self.start_idx)] + [board.__str__()
                                                         for board in self.board_list]
        str_boards = [s.split('\n') for s in str_boards]
        for row in range(2*BOARD_SIZE):
            if row == BOARD_SIZE:
                midstring = ''
                for time in range(len(str_boards)):
                    midtime = '   t' + str(time) + ': '
                    while len(midtime) < interspace:
                        midtime = ' ' + midtime
                    midstring += midtime
                    midstring += str_boards[time][row]

                s += midstring
            else:
                s += (' '*interspace) + (' '*interspace).join([str_board[row] for str_board in str_boards])
            s += '\n'
        return s


class Multiverse:
    def __init__(self, main_timeline: Timeline, up_list: [Timeline] = None, down_list: [Timeline] = None):
        self.main_timeline = main_timeline
        if up_list is None:
            up_list = []
        if down_list is None:
            down_list = []
        self.up_list = up_list
        self.down_list = down_list
        self.max_length = None
        self._set_max_length()

    def _set_max_length(self):
        self.max_length = self.main_timeline.end_time() + 1
        for listt in self.up_list, self.down_list:
            for timeline in listt:
                self.max_length = max(self.max_length, timeline.end_time() + 1)

    def clone(self):
        return Multiverse(
            main_timeline=self.main_timeline.clone(),
            up_list=[None if timeline is None else timeline.clone() for timeline in self.up_list],
            down_list=[None if timeline is None else timeline.clone() for timeline in self.down_list],
        )

    def flip_multiverse(self):
        self.main_timeline.flip_timeline()
        self.up_list, self.down_list = self.down_list, self.up_list
        for listt in self.up_list, self.down_list:
            for timeline in listt:
                if timeline is not None:
                    timeline.flip_timeline()

    def get_board(self, td_idx) -> Board | None:
        time_idx, dim_idx = td_idx
        timeline = self.get_timeline(dim_idx=dim_idx)
        if timeline is None:
            return None
        else:
            return timeline.get_board(time_idx)

    def add_board(self, td_idx, board):
        """
        adds board at specified td_idx
        :param board:
        :return:
        """
        time_idx, dim_idx = td_idx
        if dim_idx == 0:
            self.main_timeline.append(board=board, time_idx=time_idx)
            self.max_length = max(self.max_length, self.main_timeline.end_time() + 1)
            return

        if dim_idx > 0:
            dim_idx, listt = dim_idx - 1, self.up_list
        else:
            dim_idx, listt = -dim_idx - 1, self.down_list

        if dim_idx >= len(listt):
            # this should only need to happen once
            timeline = Timeline(start_idx=time_idx)
            listt.append(timeline)
        else:
            timeline = listt[dim_idx]
        timeline.append(board=board, time_idx=time_idx)
        self.max_length = max(self.max_length, timeline.end_time() + 1)

    def get_range(self):
        """
        returns range of indices (inclusive)
        """
        return (-len(self.down_list), len(self.up_list))

    def in_range(self, dim):
        bot, top = self.get_range()
        return bot <= dim and dim <= top

    def idx_exists(self, td_idx):
        time, dim = td_idx
        return self.in_range(dim) and self.get_timeline(dim).in_range(time)

    def leaves(self):
        """
        where did you go
        :return: coordinates of all final states

        WILL ALWAYS RETURN COORDS IN SAME ORDER
        """
        overall_range = self.get_range()
        for dim_idx in range(overall_range[0], overall_range[1] + 1):
            yield (self.get_timeline(dim_idx=dim_idx).end_time(), dim_idx)

    def get_timeline(self, dim_idx) -> Timeline | None:
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
            return self.main_timeline

    def remove_board(self, dim):
        """
        removes last board at specified dimension, removes timeline if no longer exists
        """
        if self.in_range(dim):
            if dim > 0:
                self.up_list[dim - 1].pop()
                while self.up_list and self.up_list[-1].is_empty():
                    self.up_list.pop()
            elif dim < 0:
                self.down_list[-dim - 1].pop()
                while self.down_list and self.down_list[-1].is_empty():
                    self.down_list.pop()
            else:
                self.main_timeline.pop()
            self._set_max_length()

    def compressed(self):
        return (
            self.main_timeline.compressed(),
            tuple(timeline.compressed() for timeline in self.up_list),
            tuple(timeline.compressed() for timeline in self.down_list),
        )

    @staticmethod
    def decompress(compression):
        compressed_main, compressed_up_list, compressed_down_list = compression
        return Multiverse(
            main_timeline=Timeline.decompress(compressed_main),
            up_list=[Timeline.decompress(comp_down) for comp_down in compressed_up_list],
            down_list=[Timeline.decompress(comp_down) for comp_down in compressed_down_list],
        )

    def __str__(self):
        s = ''
        overall_range = self.get_range()
        for dim in range(overall_range[1], overall_range[0] - 1, -1):
            timeline = self.get_timeline(dim_idx=dim)
            s += 'dimension ' + str(dim) + ':\n'
            s += timeline.__str__()
            s += '\n\n'
        return s


class Chess5d:
    def __init__(self,
                 initial_multiverse=None,
                 initial_timeline=None,
                 initial_board=None,
                 first_player=0,
                 check_validity=False,
                 save_moves=True):
        """
        implemented 5d chess
        tries to initalize like this:
            if initial_multiverse exists, uses this
            if intial_timeline exists, makes a multiverse with just this timeline
            if initial_board exists, makes a timeline with jsut this board
            otherwise, default board

        :param first_player: which player plays on the first board, default 0 (white)
        :param check_validity: whether to run validity check on each move; default false
        :param save_moves: whether to save moves (useful for undoing moves); default True

        NOTE:
            THE WAY TO TEST FOR CHECKMATE/STALEMATE SHOULD BE THE FOLLOWING:
                if the current player cannot advance the present with any set of moves, stalemate
                if the current player captures opponent king:
                    IF on the opponents last turn, there was a king in check:
                        checkmate
                    OTHERWISE:
                        IF on the opponents last turn, there was no possible sequence of moves to avoid this:
                            stalemate
                        OTHERWISE:
                            checkmate

            the reason we do this is because a 'stalemate test' is expensive
                we have to check all possible sequences of opponent moves
                thus, we do this check maybe once at the end of each game to help runtime
        """
        if initial_multiverse is None:
            if initial_timeline is None:
                if initial_board is None:
                    initial_board = Board(player=first_player)
                initial_timeline = Timeline(board_list=[initial_board])
            initial_multiverse = Multiverse(main_timeline=initial_timeline)
        self.check_validity = check_validity
        self.save_moves = save_moves
        self.multiverse = initial_multiverse
        self.first_player = first_player
        self.move_history = []
        self.dimension_spawn_history = dict()
        self.passed_boards = []

    def clone(self):
        game = Chess5d(initial_multiverse=self.multiverse.clone(),
                       first_player=self.first_player,
                       check_validity=self.check_validity,
                       save_moves=self.save_moves,
                       )
        game.move_history = copy.deepcopy(self.move_history)
        game.dimension_spawn_history = copy.deepcopy(self.dimension_spawn_history)
        return game

    @staticmethod
    def flip_moves(moves):
        return [Chess5d._flip_move(move) for move in moves]

    @staticmethod
    def _flip_move(move):
        if move == END_TURN:
            return move
        ((time1, dim1, i1, j1), (time2, dim2, i2, j2)) = move
        return ((time1, -dim1, BOARD_SIZE - 1 - i1, j1), (time2, -dim2, BOARD_SIZE - 1 - i2, j2))

    def flip_game(self):
        self.first_player = 1 - self.first_player
        self.multiverse.flip_multiverse()
        self.move_history = Chess5d.flip_moves(self.move_history)
        self.dimension_spawn_history = {((time_idx, -dim_idx), Chess5d._flip_move(move)): -dim
                                        for ((time_idx, dim_idx), move), dim in self.dimension_spawn_history.items()}

    def get_active_number(self, dim_range=None):
        """
        returns number of potential active timelines in each direction

        i.e. if return is 5, any timeline at most 5 away is active
        """
        if dim_range is None:
            dim_range = self.multiverse.get_range()
        return min(dim_range[1], -dim_range[0]) + 1

    def dim_is_active(self, dim):
        return abs(dim) <= self.get_active_number()

    def make_move(self, move):
        """
        moves piece at idx to end idx
        :param move = (idx, end_idx)
            idx: (time, dimension, x, y), must be on an existing board
            end_idx: (time, dimension, x, y), must be to an existing board, and
        this will edit the game state to remove the piece at idx, move it to end_idx
        :return (piece captured (empty if None), terminal move)
            NOTE: terminal move does NOT check for stalemate
            it simply checks if either a king was just captured, or the player has no moves left
        """

        if move == END_TURN:
            if self.save_moves:
                self.move_history.append(move)
            return EMPTY, self.no_moves(player=self.player_at(self.present()))

        idx, end_idx = move
        if self.check_validity:
            if not self.board_can_be_moved(idx[:2]):
                raise Exception("WARNING MOVE MADE ON INVALID BOARD: " + str(idx[:2]))
            if end_idx not in list(self.piece_possible_moves(idx)):
                raise Exception("WARNING INVALID MOVE: " + str(idx) + ' -> ' + str(end_idx))
        time1, dim1, i1, j1 = idx

        if self.move_history and self.last_player_moved() != self.player_at(time=time1):
            # the previous player did not end their turn, so we can just do it here instead
            self.make_move(END_TURN)
        if not self.move_history and self.first_player != self.player_at(time=time1):
            # it is the first move, and the player that moved is not the first player
            self.make_move(END_TURN)

        old_board = self.get_board((time1, dim1))
        new_board, piece = old_board.remove_piece((i1, j1))
        if piece == EMPTY:
            raise Exception("WARNING: no piece on square")
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
        else:
            # this is the timeline that piece left behind
            new_board.depassant(just_moved=None)  # there are no just moved pawns

            self.add_board_child((time1, dim1), new_board, move=move)
            newer_board, capture = self.get_board((time2, dim2)).add_piece(piece, (i2, j2))

            # even if this is a pawn, enpassant is not possible with timespace jumps
            newer_board.depassant(just_moved=None)
            self.add_board_child((time2, dim2), newer_board, move=move)
        player = self.player_at(time=time1)
        terminal = piece_id(capture) == KING or self.no_moves(player=player)

        # this is a terminal move if we just captured the king, or if the player that just moved has no moves
        # we know the player that just moved has the next move since the last move was not END_TURN
        if self.save_moves:
            self.move_history.append(move)
        return capture, terminal

    def dimension_made_by(self, td_idx, move):
        return self.dimension_spawn_history[(td_idx, move)]

    def undo_move(self):
        """
        undoes last move

        returns move, caputure
            (None, None) if no moves were made
        """
        if not self.move_history:
            print('WARNING: no moves to undo')
            return (None, None)
        move = self.move_history.pop()
        if move == END_TURN:
            return (move, EMPTY)
        idx, end_idx = move

        time1, dim1, i1, j1 = idx
        time2, dim2, i2, j2 = end_idx

        # no matter what, a new board is created when the piece moves from the board at (time1, dim1)
        dim = self.dimension_made_by((time1, dim1), move)
        self.multiverse.remove_board(dim)  # should be last board on dim, right after time1

        if (time1, dim1) != (time2, dim2):
            # if there is a time dimension jump, another board is created
            dim = self.dimension_made_by((time2, dim2), move)
            self.multiverse.remove_board(dim)  # should be last board on dim, right after time2

        return (move, self.get_piece(end_idx))

    def undo_turn(self, player=None):
        """
        undoes entire turn of the specified player
            if player is None, use last player moved

        always results in END_TURN being the last thing on move history, unless move history is empty
        returns (player specified, iterable of (move, captured piece) in correct order)
        """
        if not self.move_history:
            return None, None
        move = self.move_history[-1]
        while move == END_TURN:
            self.undo_move()
            if not self.move_history:
                return None, None
            move = self.move_history[-1]

        turn_history = []

        start_idx, end_idx = move
        if player is None:
            player = self.player_at(time=start_idx[0])

        while move != END_TURN and self.player_at(move[0][0]) == player:
            turn_history.append(self.undo_move())
            if not self.move_history:
                self.make_move(END_TURN)
                return player, tuple(turn_history[::-1])
            move = self.move_history[-1]
        if not move == END_TURN:
            self.make_move(END_TURN)
        return player, tuple(turn_history[::-1])  # reverse the history, since we put the moves on in reversed order

    def get_board(self, td_idx):
        return self.multiverse.get_board(td_idx)

    def get_piece(self, idx):
        (time, dim, i, j) = idx
        board = self.get_board((time, dim))
        if board is not None:
            return board.get_piece((i, j))

    def idx_exists(self, td_idx, ij_idx=(0, 0)) -> bool:
        i, j = ij_idx
        if i < 0 or j < 0 or i >= BOARD_SIZE or j >= BOARD_SIZE:
            return False
        return self.multiverse.idx_exists(td_idx=td_idx)

    def add_board_child(self, td_idx, board, move):
        """
        adds board as a child to the board specified by td_idx
        :param td_idx: (time, dimension)
        :param board: board to add
        :param move: move that created this board (for undo purposes)
        """
        time, dim = td_idx
        new_player = 1 - self.player_at(time=time)  # other players move on the added board
        board.set_player(new_player)

        if not self.idx_exists((time + 1, dim)):
            # in this case dimension does not change
            new_dim = dim
        else:
            player = self.player_at(time=time)
            overall_range = self.multiverse.get_range()
            if player == 0:  # white move
                new_dim = overall_range[0] - 1
            else:  # black move
                new_dim = overall_range[1] + 1

        self.multiverse.add_board((time + 1, new_dim), board)

        if self.save_moves:
            self.dimension_spawn_history[(td_idx, move)] = new_dim

    def board_can_be_moved(self, td_idx):
        """
        returns if the specified board can be moved from
        """
        time, dim = td_idx
        # the board itself must exist
        # if the next board exixts, a move has been made, otherwise, no move was made
        return self.idx_exists((time, dim)) and (not self.idx_exists((time + 1, dim)))

    def boards_with_possible_moves(self):
        """
        iterable of time-dimension coords of boards where a piece can potentially be moved

        equivalent to the leaves of the natural tree

        WILL ALWAYS RETURN BOARDS IN SAME ORDER
        """
        for td_idx in self.multiverse.leaves():
            yield td_idx

    def players_boards_with_possible_moves(self, player):
        """
        WILL ALWAYS RETURN BOARDS IN SAME ORDER
        """
        for (t, d) in self.boards_with_possible_moves():
            if self.player_at(time=t) == player:
                yield (t, d)

    def piece_possible_moves(self, idx, castling=True):
        """
        returns possible moves of piece at idx
        :param idx: (time, dim, i, j)
        :return: iterable of idx candidates

        WILL ALWAYS RETURN MOVES IN SAME ORDER
        """
        idx_time, idx_dim, idx_i, idx_j = idx
        piece = self.get_board((idx_time, idx_dim)).get_piece((idx_i, idx_j))
        pid = piece_id(piece)
        q_k_dims = itertools.chain(*[itertools.combinations(range(4), k) for k in range(1, 5)])

        if pid in (ROOK, BISHOP, QUEEN, KING, UNICORN, DRAGON, PRINCESS):  # easy linear moves
            if pid == ROOK:
                dims_to_change = itertools.combinations(range(4), 1)
            elif pid == BISHOP:
                dims_to_change = itertools.combinations(range(4), 2)
            elif pid == UNICORN:
                dims_to_change = itertools.combinations(range(4), 3)
            elif pid == DRAGON:
                dims_to_change = itertools.combinations(range(4), 4)
            elif pid == PRINCESS:
                # combo of rook and bishop
                dims_to_change = itertools.chain(itertools.combinations(range(4), 1),
                                                 itertools.combinations(range(4), 2))
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
            player = self.player_at(time=idx_time)
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
        if castling and pid == KING and is_unmoved(piece):
            # for rook_i in (0, BOARD_SIZE - 1):
            rook_i = idx_i  # rook must be on same rank
            for rook_j in (0, BOARD_SIZE - 1):
                # potential rook squares
                rook_maybe = self.get_piece((idx_time, idx_dim, rook_i, rook_j))
                if player_of(rook_maybe) == player_of(piece):
                    if piece_id(rook_maybe) == ROOK and is_unmoved(rook_maybe):
                        dir = np.sign(rook_j - idx_j)
                        works = True

                        rook_dist = abs(rook_j - idx_j)
                        # if there is a piece blocking a middle square, castling this way is bad
                        if any(player_of(self.get_piece((idx_time, idx_dim, rook_i, idx_j + dir*k)))
                               is not None for k in range(1, rook_dist)):
                            continue

                        king_hop_squares = {(idx_time, idx_dim, rook_i, idx_j + dir*k) for k in range(1, 3)}
                        # if there is a piece attacking a square the king hops over
                        # for some reason time travel is ignored
                        for square in self.attacked_squares(player=1 - player_of(piece), time_travel=False):
                            if square in king_hop_squares:
                                works = False
                                break
                        if works:
                            yield (idx_time, idx_dim, idx_i, idx_j + 2*dir)

    def board_all_possible_moves(self, td_idx, castling=True):
        """
        WILL ALWAYS RETURN MOVES IN SAME ORDER
        """
        t, d = td_idx
        board = self.get_board(td_idx)
        for (i, j) in board.pieces_of(self.player_at(t)):
            idx = (t, d, i, j)
            for end_idx in self.piece_possible_moves(idx, castling=castling):
                yield idx, end_idx

    def all_possible_moves(self, player=None, castling=True):
        """
        returns an iterable of all possible moves of the specified player
        if player is None, uses the first player that needs to move
        None is included if the player does not NEED to move

        WILL ALWAYS RETURN MOVES IN SAME ORDER
        """
        if player is None:
            player = self.player_at(time=self.present())
        if self.player_at(time=self.present()) != player:
            # the player does not have to move
            yield END_TURN
        for td_idx in self.players_boards_with_possible_moves(player=player):
            for move in self.board_all_possible_moves(td_idx=td_idx, castling=castling):
                yield move

    def all_possible_turn_subsets(self, player=None):
        """
        returns an iterable of all possible turns subsets of the specified player
            if player is None, uses the first player that needs to move

        this problem is equivalent to the following:
            consider the set of all possible moves, ignoring those that do not move to a currently active board
                (i.e. ignore any moves that always spawn a new timeline)
            break these into equivalence classes by their start/end dimensions
            DAG subgraphs in this graph where each vertex is the source of at most one edge are exactly equivalent to
                all subsets of moves that are possible. (we can additionally add back any of the ignored moves so that
                    each board is the source of at most one move)

            first, for any DAG subgraph of this type we can always do all of the moves
                    this is true since we can topologically sort them then do them from the bottom up
                    this works bc a move is impossible if the board it starts from has already been moved to
                    since we are going backwards through the DAG, each starting board cannot have been moved to
            additionally, we can add any subset of the ignored moves to this set of moves and have the same property
                    this is true since we can just do all the ignored moves first.
                        since no ignored move ends at a potential 'starting' board, this will not interfere with anything
            Thus, the moves corresponding to a DAG subgraph of the graph where each vertex is the source of one edge
                unioned with any of the ignored moves is a valid set of moves

            for the other direction, consider a valid set of moves. Since they are valid, they must have some order
                that allows them to be valid in the game
            This order corresponds to a topological sort of the edges in our constructed directed graph. This proves
                that the graph is a DAG since in the game, a board that is the end of a move cannot start a new move
                additionally, each vertex is the source of exactly one edge, as there is only one move per board

        NOTE: it is not true that the result of a set of moves are independent of the order
            if we spawn two new timelines, the order we did the moves determines the numbers of the timelines
            thus, order does matter for the result of the moves

        Thus, for this method, we eventually have to check all possible permutations of these edges

        NOTE: THIS IS ONLY NEEDED TO CALL ONCE PER GAME AS A STALEMATE CHECK,
            TO CHECK IF A CAPTURE OF A KING COULD HAVE BEEN AVOIDED

            PLAYING THE GAME LIKE THIS WILL BE BETTER FOR RUNTIME
        """

        def all_DAG_subgraphs_with_property(edge_list: dict, used_sources=None, used_vertices=None):
            """
            iterable of all lists of edges that create a DAG such that each vertex is the source of at most one edge
                (we run through all permutations, sort of wasteful)
            the order returned will be in reverse topological order (the correct traversal)

            :param edge_list: dict(vertex -> vertex set), must be copyable
            :return: iterable of (list[(start vertex, end vertex)], used vertices)
            """
            if used_sources is None:
                used_sources = set()
            if used_vertices is None:
                used_vertices = set()
            yield (), used_sources, used_vertices
            for source in edge_list:
                if source not in used_vertices:
                    for end in edge_list[source]:
                        for (subsub, all_source, all_used) in all_DAG_subgraphs_with_property(edge_list=edge_list,
                                                                                              used_sources=
                                                                                              used_sources.union(
                                                                                                  {source}),
                                                                                              used_vertices=
                                                                                              used_vertices.union(
                                                                                                  {source, end}),
                                                                                              ):
                            yield (((source, end),) + subsub,
                                   all_source, all_used)

        if player is None:
            player = self.player_at(self.present())

        # we will make a graph as in the description
        # this does not change theoretic runtime, but will in practice probably help a lot

        possible_boards = set(self.players_boards_with_possible_moves(player=player))

        # this will be a list of lists
        # all moves on each board playable board
        partition = dict()  # partitions all moves by start, end board
        edge_list = {td_idx: set() for td_idx in
                     possible_boards}  # edges we care about, just the edges between active boards
        non_edges = {td_idx: set() for td_idx in possible_boards}  # other moves, active -> inactive boards
        for td_idx in possible_boards:
            for move in self.board_all_possible_moves(td_idx=td_idx):
                start_idx, end_idx = move
                end_td_idx = end_idx[:2]
                if end_td_idx in possible_boards and end_td_idx != td_idx:
                    edge_list[td_idx].add(end_td_idx)
                else:
                    non_edges[td_idx].add(end_td_idx)
                equiv_class = (td_idx, end_td_idx)
                if equiv_class not in partition:
                    partition[equiv_class] = set()
                partition[equiv_class].add(move)
        for edges, used_sources, used_boards in all_DAG_subgraphs_with_property(edge_list=edge_list):
            # we must also add moves from the non-edges
            other_boards = possible_boards.difference(used_sources)
            for i in range(len(other_boards) + 1):
                for other_initial_boards in itertools.combinations(other_boards, i):
                    # other initial boards is a list of td_idxes
                    for other_final_boards in itertools.product(
                            *(non_edges[td_idx] for td_idx in other_initial_boards)):
                        other_edges = tuple(zip(other_initial_boards, other_final_boards))
                        # this is now a list of (td_idx start, td_idx end)

                        # this is now a list of (td_idx start, td_idx end) in correct order
                        order = other_edges + edges
                        # we will now choose moves of each class, and send them out
                        for move_list in itertools.product(*(partition[equiv_class] for equiv_class in order)):
                            yield move_list

    def all_possible_turn_subsets_bad(self, player=None):
        seen = set()
        yield ()
        for moveset in itertools.product(*(self.board_all_possible_moves(td_idx=td_idx)
                                           for td_idx in self.players_boards_with_possible_moves(player=player))):
            for order in itertools.permutations(moveset):
                temp_game = self.clone()
                seq = []
                for move in order:
                    if not temp_game.board_can_be_moved(move[0][:2]):
                        break

                    temp_game.make_move(move)
                    seq.append(move)
                    tup = tuple(seq)

                    if tup not in seen:
                        yield tup
                    seen.add(tup)

    def all_possible_turn_sets(self, player=None):
        """
        returns all sets of moves that player can take in order to advance present
        """

        if player is None:
            player = self.player_at(self.present())
        sign = 2*player - 1  # direction of dimensions, 1 for black, -1 for white
        possible_boards = set(self.players_boards_with_possible_moves(player=player))
        possible_presents = set(self.boards_with_possible_moves())

        for moves in self.all_possible_turn_subsets(player=player):
            possible_gifts = possible_presents.copy()  # keep track of possible presents

            dim_range = list(self.multiverse.get_range())
            used_dims = set()

            for (_, dim1, _, _), (time2, dim2, _, _) in moves:
                used_dims.add(dim1)

                if dim2 in used_dims or (time2, dim2) not in possible_boards:
                    # here, we split
                    dim_range[player] += sign
                    possible_gifts.add((time2 + 1, dim_range[player]))
                    used_dims.add(dim_range[player])
                else:
                    used_dims.add(dim2)
            active_number = self.get_active_number(dim_range=dim_range)
            present = min(time for time, dim in possible_gifts if abs(dim) <= active_number)
            success = True
            for time, dim in possible_boards:
                if dim not in used_dims:
                    if abs(dim) <= active_number:
                        if time <= present:
                            success = False
                            break

            if success:
                yield moves

    def all_possible_turn_sets_bad(self, player=None):
        seen = set()
        if self.player_at(self.present()) != player:
            yield ()
        for moveset in itertools.product(*(self.board_all_possible_moves(td_idx=td_idx)
                                           for td_idx in self.players_boards_with_possible_moves(player=player))):
            for order in itertools.permutations(moveset):
                temp_game = self.clone()
                seq = []
                for move in order:
                    if not temp_game.board_can_be_moved(move[0][:2]):
                        break

                    temp_game.make_move(move)
                    seq.append(move)
                    tup = tuple(seq)

                    if tup not in seen:
                        if temp_game.player_at(temp_game.present()) != player:
                            yield tup
                    seen.add(tup)

    def no_moves(self, player=None):
        """
        returns if no possible moves
        if player is None, return player that has to move
        """
        if player is None:
            return self.player_at(self.present())
        try:
            # why is there no better way to check if a generator is empty
            next(self.all_possible_moves(player=player))
            return False
        except StopIteration:
            return True

    @staticmethod
    def connections_of(idx, boardsize):
        """
        returns an iterable of all indices where a single chess piece can possibly connect to idx
            additionally returns idx, as no move is a valid move for the purpose of this method

        :param idx: an index with the caveat that the second dimension is always nonnegative
        :param boardsize: (D1,D2,D3,D4), the max dimensions of the board (each starts at zero, even D2)
        :return: iterable of indices (second dimension nonnegative)
        """
        yield idx
        # normal piece moves
        for dims in itertools.chain(*[itertools.combinations(range(4), k) for k in range(1, 5)]):
            for signs in itertools.product((-1, 1), repeat=len(dims)):
                vec = np.array((0, 0, 0, 0))
                for k, dim in enumerate(dims):
                    vec[dim] = signs[k]

                pos = idx + vec
                while np.all(pos >= 0) and np.all(pos < boardsize):
                    yield tuple(pos)
                    pos += vec
        # knight moves
        for dims in itertools.permutations(range(4), 2):
            for signs in itertools.product((-1, 1), repeat=len(dims)):
                vec = np.array((0, 0, 0, 0))
                for k, dim in enumerate(dims):
                    vec[dim] = signs[k]*(k + 1)
                pos = idx + vec
                if np.all(pos >= 0) and np.all(pos < boardsize):
                    yield tuple(pos)

    def attacked_squares(self, player, time_travel=True):
        """
        returns an iterable of all squares attacked by player (with repeats, as there is no fast way to filter these)
        :param time_travel: whether or not to consider time travel (sometimes this is false for castling and stuff)
        """
        # we do not want castling as you cannot capture a piece with that
        # this also causes an infinite loop, as to check for castling, we must use this method
        for move in self.all_possible_moves(player=player, castling=False):
            if move != END_TURN:
                start_idx, end_idx = move
                if time_travel == False and start_idx[:2] == end_idx[:2]:
                    # in this case, we ignore since time travel is not considered
                    continue
                yield end_idx

    def pass_all_present_boards(self):
        """
        does a PASS_TURN on all present boards
        """
        present = self.present()

        for td_idx in self.boards_with_possible_moves():
            time, dim = td_idx
            if time == present and self.dim_is_active(dim=dim):
                self.add_board_child(td_idx=td_idx, board=self.get_board(td_idx), move=PASS_TURN)
                self.passed_boards.append((time + 1, dim))

    def undo_passed_boards(self):
        """
        undoes all the PASS_TURNs
        """
        for time, dim in self.passed_boards:
            self.multiverse.remove_board(dim)
        self.passed_boards = []

    def current_player_in_check(self, player=None):
        """
        returns if current player is in check

        first, the current player must 'pass' on all present boards

        then, this iterates over all opponent attacked squares and returns if any of them are a king
            this is consistent with the definition of check, as a capture of ANY opponent king is a win
        """
        current_player = self.player_at(self.present())
        if player is not None and current_player != player:
            raise Exception("CHECKING CHECK CALLED ON PLAYER WHOSE TURN IT IS NOT")
        self.pass_all_present_boards()
        for idx in self.attacked_squares(player=1 - current_player):
            if piece_id(self.get_piece(idx=idx)) == KING:
                # if player_of(self.get_piece(idx=idx))==player
                # this check is unnecessary, as opponent cannot move on top of their own king
                self.undo_passed_boards()
                return True
        self.undo_passed_boards()
        return False

    def current_player_can_win(self):
        """
        returns if current player can win
            i.e. if a king is capturable by the current player on the next move
        """
        current_player = self.player_at(self.present())
        for idx in self.attacked_squares(player=current_player):
            if piece_id(self.get_piece(idx=idx)) == KING:
                return True
        return False

    def is_checkmate_or_stalemate(self, player=None):
        """
        returns if it is (checkmate or stalemate) for specified player
            if player is None, uses currnet player
        checks if for all possible moves of the player, the player still ends up in check
        note that this works in the case where player has no moves as well
        """
        if player is None:
            player = self.player_at(self.present())

        for moveset in self.all_possible_turn_sets(player=player):
            # TODO: WE DO NEED TO PERMUTE, BUT IS THERE A WAY TO DO THIS LATER?
            # FOR THE MOST PART, PERMUTATIONS SHOULD NOT HELP GETTING OUT OF CHECK

            # for moves in itertools.permutations(moveset):
            for moves in (moveset,):  # incorrect technically, for correctness use earlier line
                temp_game = self.clone()
                failed = False
                for move in moves:
                    # if we are permuting, it is possible the permutation is invalid. in this case, we do not need to
                    # inspect if the player is in check
                    if temp_game.board_can_be_moved(move[0][:2]):
                        temp_game.make_move(move)
                    else:
                        failed = True
                        break
                if failed:
                    break
                if not temp_game.current_player_can_win():
                    # then there exists a sequence of moves that keeps the player out of check
                    return False
        return True

    def last_player_moved(self):
        for i in range(len(self.move_history) - 1, -1, -1):
            move = self.move_history[i]
            if move != END_TURN:
                return self.player_at(move[0][0])

    def terminal_eval(self, mutation=True):
        """
        to be run if game has just ended (king was captured, or no moves left)
        :param mutation: whether to let game state be mutated
        :return: 1 if white (player 0) won, 0 if draw, -1 if black won
        """
        if mutation:
            game = self
        else:
            game = self.clone()

        last_player = game.last_player_moved()
        if game.move_history[-1] == END_TURN:
            last_player = 1 - last_player  # last player is the next player, as the last player that moved ended turn
            last_turn = ()
        else:
            _, last_turn = game.undo_turn()

        no_moves_left = game.no_moves(player=last_player)  # if the last player had remaining moves

        king_captured = KING in {piece_id(captured) for _, captured in last_turn}
        if king_captured:
            # here the king was captured so either the last player won or drew
            # if the opponent (previous turn) was in check, this is a win
            # if the opponent (previous turn) was not in check:
            #   if there was a turn that got opponent out of check, this is a loss for opponent (win for last player)
            #   otherwise, a draw

            # opponent=1-last_player
            opponent, previous_turn = game.undo_turn()
            if game.current_player_in_check():
                # loss for opponent, win for last player
                # if last player was white (0), this is a 1
                # otherwise 0
                return 1 - 2*last_player
            else:
                if game.is_checkmate_or_stalemate(player=opponent):
                    # opponent was not in check, and all moves led to check, so this is stalemate
                    return 0
                else:
                    # there was a move to get out of check, it was unfound, so opponent lost
                    return 1 - 2*last_player
        else:
            if not no_moves_left:
                raise Exception("TERMINAL EVAL CALLED ON NON-TERMINAL STATE")
            # here the king was not captured, so last player must have had no moves left
            # either a loss for last player (there was a valid turn that doesnt lead to check)
            # or a draw (all valid turns lead to check)
            if game.is_checkmate_or_stalemate(player=last_player):
                # there are no valid moves to get out of check
                if game.current_player_in_check():
                    # if last player was in check at the start of their turn, this is a loss (checkmate)
                    return 2*last_player - 1
                else:
                    # if there was no check, this is stalemate, so a draw
                    return 0
            else:
                # there was a valid sequence of moves that did not lead to check, unfound by last player
                # thus, last player lost
                return 2*last_player - 1

    def player_at(self, time=None):
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
        return min(t for (t, d) in self.boards_with_possible_moves() if self.dim_is_active(d))

    def compressed(self):
        """
        space efficient representation
        """
        game = self.clone()
        while game.move_history:
            game.undo_move()
        return (
            self.first_player,
            copy.deepcopy(self.move_history),
            game.multiverse.compressed()
        )

    @staticmethod
    def decompress(compression):
        first_player, moves, compressed_multiverse = compression
        game = Chess5d(initial_multiverse=Multiverse.decompress(compressed_multiverse), first_player=first_player)
        for move in moves:
            game.make_move(move)
        return game

    @staticmethod
    def get_input_dim():
        I, J, board_channels = Board.encoding_shape()
        return board_channels + 3

    def encoding_shape(self):
        overall_range = self.multiverse.get_range()
        dimensions = 1 + overall_range[1] - overall_range[0]
        I, J, board_channels = Board.encoding_shape()
        return (self.multiverse.max_length, dimensions, I, J, Chess5d.get_input_dim())

    def encoding(self):
        encoded = np.zeros(self.encoding_shape())
        I, J, board_channels = Board.encoding_shape()
        overall_range = self.multiverse.get_range()
        dimensions_used_by_players = np.sign(overall_range[1] + overall_range[0])
        # if negative, white used more dimensions, if postive, black used more dimensions, 0 if equal
        encoded[:, :, :, :, board_channels + 2] = dimensions_used_by_players

        for dim in range(overall_range[0], overall_range[1] + 1):
            for time in range(self.multiverse.max_length):
                board = self.get_board((time, dim))

                if board is None:
                    board_encoding = Board.blocked_board_encoding()
                else:
                    board_encoding = board.encoding()

                idx_of_dim = dim - overall_range[0]  # since these must start at 0
                encoded[time, idx_of_dim, :, :, :board_channels] = board_encoding
                encoded[time, idx_of_dim, :, :, board_channels] = self.board_can_be_moved((time, dim))
                encoded[time, idx_of_dim, :, :, board_channels + 1] = self.dim_is_active(dim)

        return encoded

    @staticmethod
    def decoding(array):
        times, dims, *_ = array.shape
        *_, board_channels = Board.encoding_shape()
        dimensions_used_by_players = array[0, 0, 0, 0, board_channels + 2]

        active_range = [0, 0]
        for dim in range(dims):
            if array[0, dim, 0, 0, board_channels + 1]:
                active_range[0] = dim
                break
        for dim in range(dims):
            if array[0, dim, 0, 0, board_channels + 1]:
                active_range[1] = dim

        center_range = copy.deepcopy(active_range)
        if dimensions_used_by_players == -1:
            center_range[0] += 1
        elif dimensions_used_by_players == 1:
            center_range[1] -= 1
        d_0 = sum(center_range)//2

        def get_timeline(dim):
            board_list = []
            start_time = 0

            for time in range(times):
                if Board.is_blocked(array[time, dim, :, :, :board_channels]):
                    start_time += 1
                else:
                    break
            for time in range(start_time, times):
                board = Board.decoding(array[time, dim, :, :, :board_channels])
                if board is None:
                    break
                board_list.append(board)
            return Timeline(board_list=board_list, start_idx=start_time)

        multiverse = Multiverse(main_timeline=get_timeline(dim=d_0),
                                up_list=[get_timeline(dim) for dim in range(d_0 + 1, dims)],
                                down_list=[get_timeline(dim) for dim in range(d_0 - 1, -1, -1)],
                                )
        game = Chess5d(initial_multiverse=multiverse)
        game.player = game.get_board((0, 0)).player
        return game

    @staticmethod
    def flip_encoding(array):
        """
        returns the array that would be the flipped game
        """
        game = Chess5d.decoding(array)
        game.flip_game()
        return game.encoding()

    def __eq__(self, other):
        return np.array_equal(self.encoding(), other.encoding())

    def __str__(self):
        return self.multiverse.__str__()


class Chess2d(Chess5d):
    def __init__(self, board=None, first_player=0, check_validity=False, save_moves=True):
        if board is not None:
            first_player = board.player
        super().__init__(
            initial_board=board,
            first_player=first_player,
            check_validity=check_validity,
            save_moves=save_moves,
        )

    def piece_possible_moves(self, idx, castling=True):
        # only return moves that do not jump time-dimensions
        for end_idx in super().piece_possible_moves(idx, castling=castling):
            if end_idx[:2] == idx[:2]:
                yield end_idx

    def material_draw(self):
        board = self.get_current_board()
        for idx in board.all_pieces():
            piece = board.get_piece(idx)
            if piece_id(piece=piece) != KING:
                return False
        return True

    def make_move(self, move):
        """
        only need i,j coords
        :param move: (idx, end_idx)
            idx: (i,j)
            end_idx: (i,j)
        """
        if move == END_TURN:
            capture, terminal = super().make_move(move)
            if self.material_draw():
                return capture, True
            else:
                return capture, terminal
        idx, end_idx = move
        if len(idx) == 4:
            capture, terminal = super().make_move(move)
        else:
            time = self.multiverse.max_length - 1
            dim = 0
            real_move = ((time, dim) + tuple(idx), (time, dim) + tuple(end_idx))
            capture, terminal = super().make_move(real_move)
        if self.material_draw():
            return capture, True
        else:
            return capture, terminal

    def terminal_eval(self, mutation=True):
        if self.material_draw():
            return 0
        else:
            return super().terminal_eval(mutation=mutation)

    def play(self):
        while True:
            print(self.__str__())
            moves = list(self.all_possible_moves())
            starters = list(set(idx for idx, _ in moves))

            for i, starter in enumerate(starters):
                print(i, ':', piece_id(self.get_piece(starter)), starter[2:])
            choice = ''
            while not choice.isnumeric() or int(choice) < 0 or int(choice) >= len(starters):
                choice = input('type which piece to move: ')
            starter = starters[int(choice)]
            finishers = list(set(end_idx for idx, end_idx in moves if idx == starter))

            for i, finisher in enumerate(finishers):
                print(i, ':', finisher[2:])
            choice = ''
            while not choice.isnumeric() or int(choice) < 0 or int(choice) >= len(finishers):
                choice = input('type where to move it: ')
            finisher = finishers[int(choice)]
            captuer = self.make_move((starter, finisher))
            if piece_id(captuer) == KING:
                break

    def get_current_board(self) -> Board:
        return self.multiverse.get_board((self.multiverse.max_length - 1, 0))

    @staticmethod
    def get_input_dim():
        I, J, board_channels = Board.encoding_shape()
        return board_channels

    def encoding_shape(self):
        I, J, board_channels = Board.encoding_shape()
        return (1, 1, I, J, Chess2d.get_input_dim())

    def encoding(self):
        encoded = np.zeros(self.encoding_shape())
        encoded[0, 0, :, :, :] = self.get_current_board().encoding()
        return encoded

    @staticmethod
    def decoding(array):
        board = Board.decoding(array.view(Board.encoding_shape()))
        game = Chess2d(board=board, first_player=board.player)
        return game

    def compressed(self):
        """
        space efficient representation
        """
        game = self.clone()
        while game.move_history:
            game.undo_move()
        timeline = game.multiverse.get_timeline(0)
        end_time = timeline.end_time()
        first_player = self.player_at(end_time)
        first_board = timeline.get_board(end_time)
        return (
            first_player,
            copy.deepcopy(self.move_history),
            first_board.compressed()
        )

    @staticmethod
    def decompress(compression):
        first_player, moves, compressed_board = compression
        board = Board.decompress(compressed_board)
        game = Chess2d(board=board, first_player=first_player)
        for move in moves:
            game.make_move(move)
        return game

    def clone(self):
        game = Chess2d(
            check_validity=self.check_validity,
            save_moves=self.save_moves,
        )
        game.first_player = self.first_player
        game.multiverse = self.multiverse.clone()
        game.move_history = copy.deepcopy(self.move_history)
        game.dimension_spawn_history = copy.deepcopy(self.dimension_spawn_history)
        return game

    def __str__(self):
        return self.get_current_board().__str__()


if __name__ == '__main__':
    game = Chess5d(save_moves=True, check_validity=True)
    gameclone = game.clone()
    moves = [
        ((0, 0, 1, 3), (0, 0, 3, 3)),
        'END_TURN',
        ((1, 0, 6, 4), (1, 0, 4, 4)),
        'END_TURN',
        ((2, 0, 0, 1), (0, 0, 2, 1)),
        'END_TURN',
        ((1, -1, 6, 6), (1, -1, 5, 6)),
        'END_TURN',
        ((2, -1, 1, 7), (2, -1, 2, 7)),
        'END_TURN',
        ((3, 0, 6, 6), (3, -1, 6, 6)),
        'END_TURN',
        ((4, -1, 2, 1), (4, 0, 4, 1)),
        'END_TURN',
        ((5, 0, 4, 4), (5, -1, 4, 4)),
        'END_TURN',
        ((6, 0, 4, 1), (6, -1, 4, 3)),
        'END_TURN',
        ((7, -1, 7, 1), (7, 0, 5, 1)),
        'END_TURN',
        ((8, -1, 0, 1), (8, 0, 2, 1)),
        'END_TURN',
        ((9, 0, 5, 1), (9, -1, 7, 1)),
        'END_TURN',
        ((10, 0, 2, 1), (10, -1, 0, 1)),
        'END_TURN',
        ((11, 0, 7, 3), (11, 0, 3, 7)),
        ((11, -1, 7, 1), (11, -1, 5, 2)),
        'END_TURN',
        ((12, -1, 4, 3), (8, -1, 4, 2)),
        # ((12, 0, 0, 6), (10, 0, 2, 6)),
        # 'END_TURN',
        # ((13, 0, 3, 7), (1, 0, 3, 7)),
        # 'END_TURN',
        # ((2, 1, 0, 1), (6, 0, 0, 1)),
    ]

    gameclone2 = game.clone()
    for i, move in enumerate(moves):
        if i == 5:
            gameclone2 = game.clone()
        print('capture', game.make_move(move))
    gameclone3 = game.clone()
    while len(gameclone3.move_history) > len(gameclone2.move_history):
        gameclone3.undo_move()
    assert gameclone3 == gameclone2
    for _ in range(10):
        game.undo_move()
        print('game:')
        print(game)
        encode = (game.encoding())
        game2 = Chess5d.decoding(encode)
        compression = game.compressed()
        game3 = Chess5d.decompress(compression)

        assert (game == game2)
        assert (game2 == game3)

        print('length', len(game.move_history), 'history', game.move_history)

        player = game.player_at(game.present())

        thingy = list(game.all_possible_turn_subsets(player=player))
        thingy2 = set(tuple(sorted(move)) for move in thingy)
        thingy3 = list(game.all_possible_turn_sets(player=player))
        thingy4 = set(tuple(sorted(move)) for move in thingy3)
        print(len(thingy))
        print('found subsets', len(thingy2))
        print('actual sets', len(thingy4))

        bad_thingy = list(game.all_possible_turn_subsets_bad(player=player))
        bad_thingy2 = set(tuple(sorted(move)) for move in bad_thingy)

        bad_thingy3 = list(game.all_possible_turn_sets_bad(player=player))
        bad_thingy4 = set(tuple(sorted(move)) for move in bad_thingy3)

        print()
        print(len(bad_thingy))
        print('found subsets', len(bad_thingy2))
        print('actual sets', len(bad_thingy4))
        assert (thingy2 == bad_thingy2)
        assert (thingy4 == bad_thingy4)
        # print((thingy4.difference(bad_thingy4)).pop())

    quit()
    T = 4
    connections = (Chess5d.connections_of([T//2 for _ in range(4)], [T for _ in range(4)]))
    for idx in connections:
        print(idx)
    print(len(list(connections)))
    print(T**4)
    idxs = (zip(*Chess5d.connections_of([T//2 for _ in range(4)], [T for _ in range(4)])))
    print(list(idxs))
