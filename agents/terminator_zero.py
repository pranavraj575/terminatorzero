import torch
from torch import nn
import os, pickle
import random
import time as time_module

from src.agent import Agent
from agents.mcts import UCT_search, create_pvz_evaluator
from src.chess5d import Chess5d, Chess2d, EMPTY, KING, piece_id, END_TURN
from networks.architectures import AlphaArchitecture, TransArchitect, ConvolutedArchitect, ConvolutedTransArchitect, \
    evaluate_network
from agents.replay_buffer import ReplayBuffer


class TerminatorZero(Agent):
    def __init__(self,
                 network: AlphaArchitecture,
                 memory_capacity=5000,
                 training_num_reads=1000,
                 lr=1e-3,
                 chess2d=False,
                 ):
        """

        :param network:
        :param memory_capacity:
        :param training_num_reads:
        :param lr:
        :param chess2d: whether or not this is chess 2d
            need to tweak the moves and compression a bit in this case
        """
        super().__init__()
        self.network = network
        self.buffer = ReplayBuffer(capacity=memory_capacity)
        self.training_num_reads = training_num_reads
        self.dataset = []
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.chess2d = chess2d
        if chess2d:
            self.decompressor = Chess2d.decompress
        else:
            self.decompressor = Chess5d.decompress
        self.info = {
            'epochs': 0,
            'policy loss': [],
            'value loss': [],
            'seed': 69,
        }

    def pick_move(self, game: Chess5d, player):
        moves = list(game.all_possible_moves(player=player))
        policy, value = evaluate_network(network=self.network,
                                         game=game,
                                         player=player,
                                         moves=moves,
                                         chess2d=self.chess2d)
        # return moves[torch.argmax(policy.flatten())]

        best_move, root = UCT_search(game=game,
                                     player=player,
                                     num_reads=self.training_num_reads,
                                     policy_value_evaluator=create_pvz_evaluator(self.network, chess2d=self.chess2d),
                                     )
        return best_move

    def get_losses(self, game: Chess5d,
                   player,
                   policy_target,
                   value_target,
                   policy_smoothing=0.,
                   ) -> (torch.Tensor, torch.Tensor):
        """
        :param policy_smoothing: add small amount to policy to avoid log(0)
        """
        moves = list(game.all_possible_moves(player=player))
        policy, value = evaluate_network(network=self.network,
                                         game=game,
                                         player=player,
                                         moves=moves,
                                         chess2d=self.chess2d,
                                         )

        # crossentropy loss
        # annoying to use torch.nn.CrossEntropyLoss since it assumes the output is pre softmax
        # so, just implement it manually like this
        policy_target = torch.tensor(policy_target, dtype=torch.float)
        policy_loss = -torch.dot(policy_target,
                                 torch.log(policy_smoothing + policy.flatten()))

        # calculate the entropy of the target distribution, and subtract it
        # all this does is shift the loss so that the optimal loss is 0
        # this will not affect the gradients at all, as this is a constant wrt the network params
        optimal_policy_loss = -torch.dot(policy_target, torch.log(policy_target))
        policy_loss = policy_loss - optimal_policy_loss

        value_criterion = nn.SmoothL1Loss()
        value_loss = value_criterion(value.flatten(), torch.tensor([value_target], dtype=torch.float))
        return policy_loss, value_loss

    def save_all(self, path):
        """
        saves all info to a folder
        """
        if not os.path.exists(path):
            os.makedirs(path)

        self.buffer.save(os.path.join(path, 'buffer.pkl'))
        torch.save(self.network.state_dict(), os.path.join(path, 'model.pkl'), _use_new_zipfile_serialization=False)
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pkl'), _use_new_zipfile_serialization=False)
        f = open(os.path.join(path, 'info.pkl'), 'wb')
        pickle.dump(self.info, f)
        f.close()

    def save_checkpoint(self, path, epoch, save_overall=True):
        folder = os.path.join(path, 'checkpoints', str(epoch))
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save_all(folder)
        if save_overall:
            self.save_all(path)

    def loadable(self, path):
        """
        whether a save path is loadable from
        """
        return (os.path.exists(os.path.join(path, 'info.pkl')) and
                os.path.exists(os.path.join(path, 'model.pkl')) and
                os.path.exists(os.path.join(path, 'optimizer.pkl')) and
                os.path.exists(os.path.join(path, 'buffer.pkl'))
                )

    def load_all(self, path):
        """
        loads all info from a folder
        """
        self.buffer.load(os.path.join(path, 'buffer.pkl'))
        self.network.load_state_dict(torch.load(os.path.join(path, 'model.pkl')))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pkl')))
        f = open(os.path.join(path, 'info.pkl'), 'rb')
        self.info = pickle.load(f)
        f.close()

    def load_last_checkpoint(self, path):
        """
        loads most recent checkpoint
            assumes folder name is epoch number
        """
        path = os.path.join(path, 'checkpoints')
        if not os.path.exists(path):
            return False
        best = -1
        for folder in os.listdir(path):
            check = os.path.join(path, folder)
            if os.path.isdir(check) and folder.isnumeric() and self.loadable(check):
                # self.load_all(check)
                best = max(best, int(folder))
        if best < 0:
            # checkpoint not found
            return False

        self.load_all(os.path.join(path, str(best)))
        return True

    def train(self,
              total_epochs,
              save_path,
              starting_games=None,
              draw_moves=float('inf'),
              batch_size=128,
              ckpt_freq=10,
              ):
        prev_epochs = self.info['epochs']
        for epoch in range(prev_epochs, total_epochs):
            start_time = time_module.time()
            print("EPOCH", epoch)
            if starting_games is None:
                game, player = None, 0
            else:
                game, player = random.choice(starting_games)
                game = game.clone()
                print('\tstarting game:')
                print('\t' + game.__str__().replace('\n', '\n\t'))
                print('\tstarting player', game.first_player)
            policy_loss, value_loss = self.epoch(game=game, first_player=player, draw_moves=draw_moves,
                                                 batch_size=batch_size)
            self.info['epochs'] += 1
            self.info['policy loss'].append(policy_loss.item())
            self.info['value loss'].append(value_loss.item())
            print('\ttime:', round(time_module.time() - start_time))
            print('\t(policy,value) loss:', self.info['policy loss'][-1], self.info['value loss'][-1])
            if not self.info['epochs']%ckpt_freq:
                print('\tsaving checkpoint (do not cancel)')
                self.save_checkpoint(path=save_path, epoch=self.info['epochs'])
                print('\tdone saving (can now cancel)')

    def epoch(self, game=None, first_player=0, draw_moves=float('inf'), batch_size=128) -> (torch.Tensor, torch.Tensor):
        self.add_training_data(game=game, first_player=first_player, network=self.network, draw_moves=draw_moves)
        return self.training_step(batch_size=batch_size)

    def training_step(self, batch_size=128) -> (torch.Tensor, torch.Tensor):
        batch_size = min(batch_size, len(self.buffer))
        self.optimizer.zero_grad()
        sample = self.buffer.sample(batch_size)
        total_policy_loss, total_value_loss = torch.zeros(1), torch.zeros(1)
        for namedtup in sample:
            game, player, policy_target, value_target = self.get_game_policy_value(namedtup)
            policy_loss, value_loss = self.get_losses(game=game,
                                                      player=player,
                                                      policy_target=policy_target,
                                                      value_target=value_target,
                                                      )
            total_policy_loss += policy_loss/batch_size
            total_value_loss += value_loss/batch_size
        overall_loss = total_policy_loss + total_value_loss
        overall_loss.backward()
        self.optimizer.step()
        return total_policy_loss, total_value_loss

    def get_tuple(self, compressed_game, player, policy, value):
        return ((compressed_game, player), policy, value)

    def get_game_policy_value(self, namedtup):
        ((compressed_game, player), policy, value) = namedtup.gameinfo, namedtup.policy, namedtup.value
        game = self.decompressor(compressed_game)
        return game, player, policy, value

    def add_training_data(self, game=None, first_player=0, network=None, draw_moves=float('inf')):
        if game is None:
            game = Chess5d()
        if network is None:
            network = self.network

        terminal = False
        data = []
        early_termination = False
        bored = 0
        player = first_player

        if game.no_moves(player=player):
            print("WARNING: training on game that starts with no possible moves")
            return
        first_value = None
        while not terminal:
            num_reads = self.training_num_reads + len(list(game.all_possible_moves(player=player)))
            best_move, root = UCT_search(game=game,
                                         player=player,
                                         num_reads=num_reads,
                                         policy_value_evaluator=create_pvz_evaluator(network, chess2d=self.chess2d),
                                         )
            if first_value is None:
                first_value = root.value_estimate
            policy = root.get_final_policy()
            # value=max(root.child_Q())
            value = 0
            data.append((game.compressed(), player, policy, value))

            captured, terminal = game.make_move(move=best_move)
            if best_move == END_TURN:
                player = 1 - player

            if captured != EMPTY:
                bored = 0
            else:
                bored += 1

            if bored >= draw_moves:
                early_termination = True
                break
        print('\ttraining game:')
        if self.chess2d:
            print('\t\tfinal state:\n')
            print('\t\t' + game.__str__().replace('\n', '\n\t\t'))
        print('\t\tvalue estimate:')
        print('\t\t' + str(first_value))

        # print(game.multiverse)
        print('\t\ttime size:', game.multiverse.max_length - 1)
        print('\t\tdim range:', game.multiverse.get_range())
        if early_termination:
            result = 0
        else:
            result = game.terminal_eval()
        print('\t\tresult:', result)

        for compressed_game, player, policy, value in data:
            value = (1 - 2*player)*result  # if player==0, use result, else use -result
            self.buffer.push(*self.get_tuple(compressed_game=compressed_game,
                                             player=player,
                                             policy=policy,
                                             value=value,
                                             ))


if __name__ == '__main__':
    from src.utilitites import seed_all
    import sys

    DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    save_dir = os.path.join(DIR, 'data', 'save_test')
    embedding_dim = 256
    num_residuals = 2

    seed_all()
    net = ConvolutedArchitect(input_dim=Chess5d.get_input_dim(),
                              embedding_dim=embedding_dim,
                              num_residuals=num_residuals
                              )
    agent = TerminatorZero(network=net,
                           training_num_reads=1,
                           lr=.001,
                           )

    agent.add_training_data(draw_moves=0)
    for i in range(100):
        loss = agent.training_step()
        if not i%10:
            print('loss', loss)
    agent.save_all(save_dir)

    agent2 = TerminatorZero(network=ConvolutedArchitect(input_dim=Chess5d.get_input_dim(),
                                                        embedding_dim=embedding_dim,
                                                        num_residuals=num_residuals
                                                        ), training_num_reads=1)

    agent3 = TerminatorZero(network=ConvolutedArchitect(input_dim=Chess5d.get_input_dim(),
                                                        embedding_dim=embedding_dim,
                                                        num_residuals=num_residuals
                                                        ), training_num_reads=1)
    agent2.load_all(save_dir)
    agent3.load_all(save_dir)
    game, player, policy_target, value_target = agent2.get_game_policy_value(agent2.buffer.sample(1)[0])
    encoding = torch.tensor(game.encoding(), dtype=torch.float).unsqueeze(0)

    policy2, value2 = (agent2.network.forward(encoding, moves=list(game.all_possible_moves(player=player))))
    policy3, value3 = (agent3.network.forward(encoding, moves=list(game.all_possible_moves(player=player))))
    assert torch.equal(policy2, policy3)
    assert torch.equal(value2, value3)
    for key, value in agent2.network.state_dict().items():
        print(key)
