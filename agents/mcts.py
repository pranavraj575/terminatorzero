import torch
from src.chess5d import Chess5d, END_TURN, KING, EMPTY, piece_id
import numpy as np
from src.agent import Agent, game_outcome
from agents.non_learning import Randy

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
                 terminal,
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
        self.terminal = terminal
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

    def get_final_policy(self):
        return torch.nn.Softmax(-1)(torch.tensor(self.child_Q())).flatten().detach().numpy()
        # Alphazero usesuse child number visits because apparently this is less prone to outliers
        return self.child_number_visits/np.sum(self.child_number_visits)
        # return torch.nn.Softmax()(self.child_Q())

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
        return self.terminal

    def terminal_eval(self, temp_game: Chess5d):
        result = temp_game.terminal_eval()
        # result is 1 if white won, 0 if draw, -1 if black won
        if self.player == 1:
            # in this case, flip the result
            result = -result
        return result

    def fully_expanded(self):
        return np.all(self.child_number_visits != 0)

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
        capture, terminal = game.make_move(move)
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
                                       terminal=terminal,
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
    """
    root_game = game
    root = Node(temp_game=root_game.clone(),
                player=player,
                move=DUMMY_MOVE,
                capture=EMPTY,
                parent=DummyNode(player=player),
                terminal=False,
                )
    if game.no_moves(player=player):
        return None, root
    for i in range(num_reads):
        leaf, leaf_game = root.select_leaf(game=root_game.clone())
        if leaf.is_terminal():
            leaf.backup(value_estimate=leaf.terminal_eval(temp_game=leaf_game))
        else:
            policy, value_estimate = policy_value_evaluator(
                game=leaf_game,
                player=leaf.player,
                moves=leaf.next_moves,
            )
            leaf.expand(child_priors=policy)
            leaf.backup(value_estimate=value_estimate)
    return root.next_moves[np.argmax(root.get_final_policy())], root


def create_pvz_evaluator(policy_value_net, chess2d=False):
    """
    creates a function to evaluate the game for a given player
    :param policy_value_net: (game encoding, moves) -> (policy, value)
        only works for white
    :return: (game, player, moves) -> (policy, value)
        returns the evaluation and policy for player
        policy is a np array, value is a scalar
    """

    def pvz(game: Chess5d, player, moves):
        if player == 1:
            # we should flip the game
            game.flip_game()
            # this is also necessary
            moves = Chess5d.flip_moves(moves)
        if chess2d:
            # all moves must be (0,0,i,j) in this case
            moves = [END_TURN if move == END_TURN else ((0, 0, *move[0][2:]), (0, 0, *move[1][2:]))
                     for move in moves]
        policy, value = policy_value_net(torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0),
                                         moves,
                                         )
        return policy.flatten().detach().numpy(), value.flatten().detach().item()

    return pvz


class MCTSAgent(Agent):
    """
    picks moves with a simple mcts
        would expect this does not go well
    """

    def __init__(self, num_reads=1000, playout_agent=None, draw_moves=float('inf')):
        """
        :param draw_moves: sent to the rollout agent
            number of moves without capture before a draw is declared
        """
        super().__init__()
        self.num_reads = num_reads
        if playout_agent is None:
            playout_agent = Randy()
        self.playout_agent = playout_agent
        self.draw_moves = draw_moves

    def pick_move(self, game: Chess5d, player):
        move, _ = UCT_search(game=game,
                             player=player,
                             num_reads=self.num_reads,
                             policy_value_evaluator=self.policy_value_eval,
                             )
        return move

    def policy_value_eval(self, game, player, moves):
        """
        returns the value of a game for specified player

        does this through random playout
        """
        policy = (99 + np.random.random(len(moves)))/100
        outcome, game = game_outcome(
            player=self.playout_agent,
            opponent=self.playout_agent,
            game=game,
            first_player=player,
            draw_moves=self.draw_moves,
        )
        if player == 1:
            # flip the value in this case
            outcome = -outcome
        return policy, outcome


if __name__ == '__main__':
    from src.chess5d import Board, Chess2d, BOARD_SIZE, QUEEN, as_player
    from src.utilitites import seed_all

    seed_all(1)

    easy_game = Chess2d(
        board=Board(pieces=[[as_player(KING, 0), as_player(QUEEN, 0)] + [EMPTY for _ in range(BOARD_SIZE - 2)]] +
                           [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                           [[as_player(KING, 1)] + [EMPTY for _ in range(BOARD_SIZE - 1)]]
                    ),
        check_validity=True
    )

    agent = MCTSAgent(num_reads=128, draw_moves=50)
    outcome, game = game_outcome(agent, agent, game=easy_game.clone(), draw_moves=50)
    print(game.multiverse)
    quit()
    stalemate = Chess2d(
        board=Board(pieces=[[as_player(KING, 1)] + [EMPTY for _ in range(BOARD_SIZE - 1)]] +
                           [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 4)] +
                           [[EMPTY, as_player(QUEEN, 1)] + [EMPTY for _ in range(BOARD_SIZE - 2)]] +
                           [[EMPTY for _ in range(BOARD_SIZE)]] +
                           [[as_player(KING, 0)] + [EMPTY for _ in range(BOARD_SIZE - 1)]]
                    ))

    next_move, root = UCT_search(stalemate, player=0, num_reads=128, policy_value_evaluator=evaluator)
    print(stalemate)
    print(next_move)
    root: Node
    child = root.children[next_move].children[END_TURN]
    child: Node
    stalemate.make_move(next_move)
    nexxxtmove = ((1, 0, 5, 1), (1, 0, 6, 0))
    grandchild = child.children[nexxxtmove]
    stalemate.make_move(nexxxtmove)
    grandchild: Node
    print(grandchild.terminal_eval(stalemate))
