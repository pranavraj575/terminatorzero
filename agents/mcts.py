import torch
from src.chess5d import Chess5d, END_TURN, KING, EMPTY, piece_id
from torch import nn
import numpy as np

DUMMY_MOVE = None


class DummyNode:
    def __init__(self, player):
        self.parent = None
        self.player = player
        self.child_total_value = np.zeros(1)
        self.child_number_visits = np.zeros(1)
        self.next_moves = [DUMMY_MOVE]
        self.move_idx = {DUMMY_MOVE: 0}
        self.dumb = True


class Node:
    def __init__(self,
                 temp_game: Chess5d,
                 player,
                 move,
                 capture,
                 parent=None,
                 next_moves=None,
                 exploration_constant=1.,
                 ):
        """
        save storage by not actually saving the whole game state

        :param temp_game: game at this stage (this is not saved in the node)
        :param player: player whose turn it is
        :param move: which move we got from parent
        :param capture: what piece was just captured
        :param parent: previous node
        :param next_moves: iterable of playable moves by player from the state
            ORDER MATTERS maybe
            if None, uses game.
        """
        self.player = player
        self.move = move
        self.parent = parent
        self.child_policy = {}
        self.is_expanded = False
        self.children = {}
        self.captured = piece_id(capture)
        self.dumb = False  # not a dummy node

        if next_moves is None:
            next_moves = list(temp_game.all_possible_moves(player=player))
        self.next_moves = list(next_moves)
        self.move_idx = {next_move: i for i, next_move in enumerate(self.next_moves)}

        self.child_priors = np.ones(len(next_moves))

        # value estimate for self of taking each move
        self.child_total_value = np.zeros(len(next_moves))
        self.child_number_visits = np.zeros(len(next_moves))
        self.exp_constant = exploration_constant

    def number_visits(self):
        return self.parent.child_number_visits[self.parent.move_idx[self.move]]

    def child_Q(self):
        return self.child_total_value/(1 + self.child_number_visits)

    def child_U(self):
        return self.exp_constant*np.sqrt(self.number_visits())*(self.child_priors/(1 + self.child_number_visits))

    def unexplored_indices(self):
        return np.where(self.child_number_visits == 0)[0]

    def pick_move(self):
        bestmove = self.child_Q() + self.child_U()
        if self.fully_expanded():
            return self.next_moves[np.argmax(bestmove)]
        else:
            # in this case, max of the unexplored nodes
            return self.next_moves[max(self.unexplored_indices(), key=lambda idx: self.child_priors[idx])]

    def select_leaf(self, game: Chess5d):
        """
        selects leaf of self, makes moves on game along the way
        :return: leaf, resulting game
        """
        current = self
        i = 0
        while current.is_expanded:
            best_move = current.pick_move()
            current, game = current.maybe_add_child(game=game, move=best_move)

            i += 1
        return current, game

    def add_dirichlet_noise(self, child_priors):
        """
        adds noise to a policy
        """
        child_priors = 0.75*child_priors + 0.25*np.random.dirichlet(
            np.zeros([len(child_priors)], dtype=np.float32) + 0.3)
        return child_priors

    def is_root(self):
        return self.parent == None

    def is_terminal(self):
        return len(self.next_moves) == 0 or self.captured == KING

    def terminal_eval(self, temp_game):
        if self.captured == KING:
            # this returns the evaluation for parent.player
            # self.player == parent.player in this case, as an END_TURN was not played
            # thus, if the parent captured the king without stalemate, the parent won, and this will return 1

            # however, we must do a stalemate check on the previous state
            return 1
        # otherwise, this is a stalemate and thus a draw
        return 0

    def fully_expanded(self):
        # we only need to check this as we will always go to an unexplored node if there is one available
        return self.number_visits() >= len(self.next_moves)

    def expand(self, child_priors):
        self.is_expanded = True

        if self.is_root():  # add dirichlet noise to child_priors in root node
            child_priors = self.add_dirichlet_noise(child_priors=child_priors)
        self.child_priors = child_priors

    def maybe_add_child(self, game: Chess5d, move):
        """
        makes move from current state,
            if move has not been seen, adds it to children
        returns resulting child and resulting game
            mutates game state
        """
        capture = game.make_move(move)
        if move not in self.children:
            if move is END_TURN:
                new_player = 1 - self.player
            else:
                new_player = self.player
            self.children[move] = Node(temp_game=game,
                                       player=new_player,
                                       move=move,
                                       capture=capture,
                                       parent=self,
                                       )
        return self.children[move], game

    def inc_total_value_and_visits(self, value_estimate):
        idx = self.parent.move_idx[self.move]
        self.parent.child_number_visits[idx] += 1
        self.parent.child_total_value[idx] += value_estimate

    def backup(self, value_estimate: float):
        if self.player != self.parent.player:
            # the given value estimate is the value of this child for the current player
            # the intended value estimate is the parent's estimate of all children
            # thus, we need to flip in this case, as the parent will get the opposite value of the child
            value_estimate = -value_estimate
        self.inc_total_value_and_visits(value_estimate=value_estimate)
        if not self.parent.dumb:
            self.parent.backup(value_estimate=value_estimate)


def UCT_search(game: Chess5d, player, num_reads, policy_value_evaluator):
    """
    :param game:
    :param player:
    :param num_reads:
    :param policy_value_evaluator: (game, player, moves) -> (policy,value)
        returns how good the game is for player, if player is the one whose move it is
        moves is all possible moves at that point (if None, uses game.all_possible_moves(player))
    :return:
    """
    root_game = game
    root = Node(temp_game=root_game.clone(),
                player=player,
                move=DUMMY_MOVE,
                capture=EMPTY,
                parent=DummyNode(player=player),
                )
    if root.is_terminal():
        return None, root
    for i in range(num_reads):
        leaf, game = root.select_leaf(game=root_game.clone())
        if leaf.is_terminal():
            leaf.backup(value_estimate=leaf.terminal_eval(temp_game=game))
        else:
            policy, value_estimate = policy_value_evaluator(
                game=game,
                player=leaf.player,
                moves=leaf.next_moves,
            )
            leaf.expand(child_priors=policy)
            leaf.backup(value_estimate=value_estimate)
    return root.next_moves[np.argmax(root.child_Q())], root


def create_pvz_evaluator(policy_value_net):
    """
    creates a function to evaluate the game for a given player
    :param policy_value_net: (game encoding, moves) -> (policy, value)
        only works for white
    :return: (game, player) -> (policy,value)
    """

    def pvz(game: Chess5d, player, moves):
        if player == 1:
            # we should flip the game
            game.flip_game()
        policy, value = policy_value_net(game.encoding(), moves)
        value: torch.Tensor
        return policy.flatten().detach().numpy(), value.flatten().detach().item()

    return pvz


if __name__ == '__main__':
    from src.chess5d import Board, Chess2d, BOARD_SIZE, QUEEN, ROOK, as_player
    from src.utilitites import seed_all

    seed_all(1)

    easy_game = Chess2d(
        board=Board(pieces=[[as_player(KING, 0), as_player(QUEEN, 0)] + [EMPTY for _ in range(BOARD_SIZE - 2)]] +
                           [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                           [[as_player(KING, 1)] + [EMPTY for _ in range(BOARD_SIZE - 1)]]
                    ))
    easy_game.check_validity = True
    evaluator = create_pvz_evaluator(policy_value_net=lambda enc, moves: (torch.rand(len(moves)), torch.zeros(1)))
    player = 0
    for _ in range(50):
        next_move, root = UCT_search(easy_game, player=player, num_reads=128, policy_value_evaluator=evaluator)
        for move, q, number in zip(root.next_moves, root.child_Q(), root.child_number_visits):
            print(move, q, number)

        capture = easy_game.make_move(next_move)
        if next_move is END_TURN:
            print("PLAYER SWAP")
            player = 1 - player
        else:
            print(next_move)
            print(easy_game)
        if piece_id(capture) == KING:
            break
    print(easy_game.multiverse)
    pass
