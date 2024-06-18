import os, sys

from src.chess5d import Chess2d, Board, KING, QUEEN, BOARD_SIZE, as_player, EMPTY
from src.utilitites import seed_all
from agents.terminator_zero import TerminatorZero
from networks.architectures import ConvolutedArchitect, TransArchitect
from networks.transformer import MultiHeadedAttentionSingleMove, MultiHeadedAttentionFull

if __name__ == '__main__':
    seed_all(2)
    DIR = os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))
    embedding_dim = 256
    num_blocks = 16
    num_reads = 1000
    # trans specific
    num_heads = 5
    drop_prob = .2
    positional_encoding_nums = (10, 10, 3, 3)
    attention_type = 'full'

    architecture = 'trans'
    game_name = 'small_board'

    ident = ('game_' + game_name +
             '_net_architecture_' + architecture +
             '_embedding_dim_' + str(embedding_dim) +
             '_num_blocks_' + str(num_blocks)+
             '_num_reads_' + str(num_reads)
             )
    if architecture == 'cnn':
        network = ConvolutedArchitect(input_dim=Chess2d.get_input_dim(),
                                      embedding_dim=embedding_dim,
                                      num_residuals=num_blocks,
                                      positional_encoding_nums=(0, 0, 3, 3),
                                      kernel=(1, 1, 3, 3),
                                      )
    elif architecture == 'trans':
        ident += '_num_heads_' + str(num_heads)
        ident += '_' + attention_type
        if attention_type == 'full':
            Attention = MultiHeadedAttentionFull
        elif attention_type == 'single_move':
            Attention = MultiHeadedAttentionSingleMove
        else:
            raise Exception(attention_type + ' not recognized')
        network = TransArchitect(input_dim=Chess2d.get_input_dim(),
                                 embedding_dim=embedding_dim,
                                 num_decoders=num_blocks,
                                 n_heads=num_heads,
                                 positional_encoding_nums=positional_encoding_nums,
                                 drop_prob=drop_prob,
                                 AttentionClass=Attention,
                                 )
    else:
        raise Exception('architecture ' + architecture + ' not valid string')

    if BOARD_SIZE != 8:
        ident += '_board_size_' + str(BOARD_SIZE)

    starting_games = []
    if game_name == 'queen_checkmate':
        left = (BOARD_SIZE - 2)//2
        board = Board(pieces=[[EMPTY for _ in range(left)] +
                              [as_player(KING, 0), as_player(QUEEN, 0)] +
                              [EMPTY for _ in range(BOARD_SIZE - 2 - left)]] +
                             [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE - 2)] +
                             [[EMPTY for _ in range(left)] +
                              [as_player(KING, 1)] +
                              [EMPTY for _ in range(BOARD_SIZE - 1 - left)]]
                      )
    elif BOARD_SIZE == 5 and game_name == 'small_board':
        board = Board()
    else:
        raise Exception('game name ' + game_name + ' not valid string')

    starting_games.append((Chess2d(board=board.clone()), board.player))
    board = board.flipped_board()
    starting_games.append((Chess2d(board=board.clone()), board.player))
    board.set_player(1 - board.player)
    starting_games.append((Chess2d(board=board.clone()), board.player))
    board = board.flipped_board()
    starting_games.append((Chess2d(board=board.clone()), board.player))

    agent = TerminatorZero(network=network,
                           training_num_reads=num_reads,
                           chess2d=True,
                           )

    save_dir = os.path.join(DIR, 'data', 'test_2d_chess', ident)
    if agent.load_last_checkpoint(path=save_dir):
        epochs = agent.info['epochs']
        print("loaded checkpoint with", epochs, "epochs from", save_dir)
    else:
        print('starting save to:', save_dir)
    if False:
        from agents.human import Human
        from agents.non_learning import Randy
        from src.agent import game_outcome

        game, first_player = starting_games[2]
        outcome, game = game_outcome(Human(), agent, game=game, first_player=first_player)
        if outcome == 0:
            print('draw')
        elif outcome == 1:
            print('you won')
        else:
            print('you lost')
        print(game.move_history)
        quit()
    agent.train(total_epochs=1000, save_path=save_dir, starting_games=starting_games, draw_moves=100, ckpt_freq=5)
