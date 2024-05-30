from collections import namedtuple, deque
import pickle, random

Game_Policy_Value = namedtuple('Game_Policy_Value',
                               ('gameinfo', 'policy', 'value'))


class ReplayBuffer:
    def __init__(self, capacity=10000, namedtup=Game_Policy_Value):
        self.memory = deque([], maxlen=capacity)
        self.namedtup = namedtup

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.namedtup(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def full(self):
        """
        returns whether memory is full
        """
        return len(self.memory) == self.memory.maxlen

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.memory, f)
        f.close()

    def load(self, filename):
        f = open(filename, 'rb')
        self.memory = pickle.load(f)
        f.close()

    def __len__(self):
        return len(self.memory)
