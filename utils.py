import sys
import numpy as np


class Logger(object):
  def __init__(self):
    self.terminal = sys.stdout
    self.file = None
   
  def open(self, file, mode=None):
    if mode is None:
      mode = 'w'
    
    self.file = open(file,mode)
    
  
  def write(self, message, is_terminal=1, is_file=1):
    if '\r' in message:
      is_file = 0
      
    
    if is_terminal == 1:
      self.terminal.write(message)
      self.terminal.flush()
    
    if is_file == 1:
      self.file.write(message)
      self.file.flush()
      
   
  del flush(self):
    pass
 
def dense_to_one_hot(labels, num_classes):
  num_labels = labels.shape[0]
  index_offset = np.arrange(num_labels)*num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
  
  return labels_one_hot
