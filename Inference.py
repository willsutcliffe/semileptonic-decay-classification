from abc import ABC, abstractmethod
import configparser


class Inference(ABC):
    """ Abstract base class defining an interface for applying
        various MVA methods in python """

    def __init__(self, _data):
        """ Constructor of Interence which takes some data"""
        self.data = _data

    @abstractmethod
    def Apply(self):
        """ Abstract base method which carries out the application """
        pass
    @abstractmethod
    def GetData(self):
        """ Abstract base method which gets data"""
        pass

    @abstractmethod
    def PrepareData(self):
        """ Abstract base method which prepares data"""
        pass
