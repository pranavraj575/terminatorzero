import os, sys

from src.chess5d import Chess2d, Board, KING, QUEEN, BOARD_SIZE, as_player, EMPTY
from src.utilitites import seed_all
from agents.terminator_zero import TerminatorZero
from networks.architectures import ConvolutedArchitect

if __name__ == '__main__':
    seed_all()
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    embedding_dim = 256
    num_blocks = 16
    num_reads = 100
    architecture = 'cnn'
    game_name = 'queen_checkmate'

    ident = ('game_' + game_name +
             '_net_architecture_' + architecture +
             '_embedding_dim_' + str(embedding_dim) +
             '_num_blocks_' + str(num_blocks) +
             '_num_reads_' + str(num_reads))

    save_dir = os.path.join(DIR, 'data', 'test_2d_chess', ident)
    if architecture == 'cnn':
        network = ConvolutedArchitect(input_dim=Chess2d.get_input_dim(),
                                      embedding_dim=embedding_dim,
                                      num_residuals=num_blocks,
                                      positional_encoding_nums=(0, 0, 3, 3),
                                      kernel=(1, 1, 3, 3),
                                      )
    else:
        raise Exception('architecture ' + architecture + ' not valid string')
    if game_name == 'queen_checkmate':
        left = (BOARD_SIZE - 2)//2
        board = Board(pieces=[[EMPTY for _ in range(left)] +
                              [as_player(KING,0), as_player(QUEEN, 0)] +
                              [EMPTY for _ in range(BOARD_SIZE - 2 - left)]] +
                             [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                             [[EMPTY for _ in range(left)] +
                              [as_player(KING,1)] +
                              [EMPTY for _ in range(BOARD_SIZE - 1 - left)]]
                      )
    else:
        raise Exception('game name ' + game_name + ' not valid string')

    starting_games = [
        (Chess2d(board=board.clone()), 0),
        (Chess2d(board=board.clone(), first_player=1), 1)
    ]
    agent = TerminatorZero(network=network,
                           training_num_reads=num_reads,
                           chess2d=True,
                           )
    if agent.load_last_checkpoint(path=save_dir):
        epochs = agent.info['epochs']
        print("loaded checkpoint with", epochs, "epochs from", save_dir)

    agent.train(total_epochs=100, save_path=save_dir, starting_games=starting_games, draw_moves=200, ckpt_freq=2)
