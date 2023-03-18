
from BDT.Inference import Inference
import xgboost as xgb

class XGBoostInference(Inference):
      """ Class for classifier inference when using XGBoost """

      def __init__(self,_weights,_features,_outputlabel='BDT_prediction',_data=None):
          """ Constructor of XGBoostTrainer """
          super().__init__(_data)
          self.features = _features
          self.classifier = xgb.Booster({'nthread': 4})
          self.classifier.load_model(_weights)
          self.outputlabel = _outputlabel
          if _data != None:
            self.PrepareData()

      def SetData(self,data):
          self.data=data
          self.PrepareData()

      def PrepareData(self):
          """ Pepares data for training"""
          self.ddata = xgb.DMatrix( self.data[self.features].values)


      def Apply(self):
          """  Apply the classifier to prepared data """
          self.data.loc[:,self.outputlabel] =  self.classifier.predict(self.ddata)

      def GetData(self):
          return (self.data)


