from src.chess5d import Chess5d, END_TURN
from torch import nn


class UCTNode:
    def __init__(self, game: Chess5d, player, network_fn, network_output, pre_softmax_from_output, move, parent=None):
        """
        save storage by not actually saving the whole game state

        :param game: game at this stage (this is not saved in the node)
        :param player: player whose turn it is
        :param network_fn: game -> output (value/policy info)
        :param pre_softmax_from_output: (move, network_output) -> real that can be fed to a softmax function for policy
        :param move: which move we got from parent
        :param parent: previous node
        """
        self.player = player
        self.network_output = network_output  # we want to do this calculation exactly once for each game state
        self.pre_softmax_from_output = pre_softmax_from_output
        self.move = move
        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.children_network_outputs = {}
        for move in game.all_possible_moves(player=player):
            if move is END_TURN:
                # this is the same game, no need to rerun network
                self.children_network_outputs[move] = self.network_output
            else:
                game.make_move(move)
                self.children_network_outputs[move] = network_fn(game)
                game.undo_move()

        self.child_policy = {}
        self._calculate_child_policy()

        self.child_total_value = {}
        self.child_number_visits = {}

    def _calculate_child_policy(self):
        moves, net_outputs = zip(*self.children_network_outputs.items())
        pre_softmaxes = torch.tensor([self.pre_softmax_from_output(move, network_output)
                                      for move, network_output in zip(moves, net_outputs)])
        softmaxes = nn.Softmax(-1)(pre_softmaxes)
        self.child_policy = {move: policy for move, policy in zip(moves, softmaxes)}

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value/(1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits)*(
                abs(self.child_priors)/(1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs]  # select only legal moves entries in child_priors array
        valid_child_priors = 0.75*valid_child_priors + 0.25*np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32) + 0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = [];
        c_p = child_priors
        for action in self.game.actions():  # possible actions
            if action != []:
                initial_pos, final_pos, underpromote = action
                action_idxs.append(ed.encode_action(self.game, initial_pos, final_pos, underpromote))
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        for i in range(len(child_priors)):  # mask all illegal actions
            if i not in action_idxs:
                c_p[i] = 0.0000000000
        if self.parent.parent == None:  # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def decode_n_move_pieces(self, board, move):
        i_pos, f_pos, prom = ed.decode_action(board, move)
        for i, f, p in zip(i_pos, f_pos, prom):
            board.player = self.game.player
            board.move_piece(i, f, p)  # move piece to get next board state s
            a, b = i;
            c, d = f
            if board.current_board[c, d] in ["K", "k"] and abs(
                    d - b) == 2:  # if king moves 2 squares, then move rook too for castling
                if a == 7 and d - b > 0:  # castle kingside for white
                    board.player = self.game.player
                    board.move_piece((7, 7), (7, 5), None)
                if a == 7 and d - b < 0:  # castle queenside for white
                    board.player = self.game.player
                    board.move_piece((7, 0), (7, 3), None)
                if a == 0 and d - b > 0:  # castle kingside for black
                    board.player = self.game.player
                    board.move_piece((0, 7), (0, 5), None)
                if a == 0 and d - b < 0:  # castle queenside for black
                    board.player = self.game.player
                    board.move_piece((0, 0), (0, 3), None)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)  # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = UCTNode(
                copy_board, move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:  # same as current.parent.game.player = 0
                current.total_value += (1*value_estimate)  # value estimate +1 = white win
            elif current.game.player == 0:  # same as current.parent.game.player = 1
                current.total_value += (-1*value_estimate)
            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)
