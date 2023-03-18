from abc import ABC,abstractmethod
import configparser

class Trainer(ABC):
      """ Abstract base class defining an interface for training 
          various MVA methods in python """

      def __init__(self, _data):
          """ Constructor of Trainer. The data is contrains a target variable
              with signal denoted with 1 and background 0 """
          self.data = _data

      @abstractmethod
      def train(self):
          """ Abstract base method which carries out the training """
          pass 

      @abstractmethod
      def test(self):
          """ Abstract base method which carries out the testing """
          pass 

     
      @abstractmethod
      def ROC(self):
          """ Abstract base method which generates a ROC curve fo
             for testing and training  """
          pass 
