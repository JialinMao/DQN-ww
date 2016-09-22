import numpy as np

class History:
  def __init__(self, config):
    self.cnn_format = config.cnn_format

    batch_size, history_length, self.screen_height, self.screen_width, self.depth = \
            config.batch_size, config.history_length, config.screen_height, config.screen_width, config.depth
    
    self.history = np.zeros(
            [history_length, self.screen_width, self.screen_height, self.depth], dtype=np.float32)
    ''' 
    self.history = np.zeros(
            [history_length, self.screen_width, self.screen_height], dtype=np.float32)
    '''
  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    if self.cnn_format == 'NHWC':
      return np.transpose( self.history.reshape([-1, self.screen_width, self.screen_height]), (1, 2, 0))
    else:
      return self.history.reshape([-1, self.screen_width, self.screen_height])

