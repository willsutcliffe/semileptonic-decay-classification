
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from root_pandas import read_root

class EventPreprocesser:
      """ Class which handles BDT preprocessing """

      def __init__(self, data = {}, tree =''):
          """ Constructor of BDTPreprocessor"""
          self.dfs = {}
          for key, value in data.items():
              self.dfs[key] = read_root(value, tree)
       
      def Combine(self,one,two,label,sample='Sample'):
          """ combines two data frames in the dictionary """
          self.dfs[one[0]][sample]=one[1]
          self.dfs[two[0]][sample]=two[1]
          self.dfs[label] = pd.concat([self.dfs[one[0]], self.dfs[two[0]]], ignore_index=True)
          self.dfs[label]['Index'] = self.dfs[label].index 
          del self.dfs[one[0]]
          del self.dfs[two[0]]
 

      def ApplyMask(self,mask):
          for key, value in self.dfs.items():
               self.dfs[key] = dfs[key].query(mask)
     
      def ApplyOneMask(self,mask,label):
          self.dfs[label] = dfs[label].query(mask)

      def FillNaNs(self,value):
          """ Fill any NaNs in dataframe """
          for key, value in self.dfs.items():
               self.dfs[key] = value.fillna(value)

      def RandomiseTestTrain(self,mask,label,Ntrain,seed=None):
          """ Adds a Train column initialised to 0 and sets 
              events within a given mask to having train = 1 """
          if(label not in self.dfs[label].columns):
            self.dfs[label].loc[:,'Train'] = 0 
          df = self.dfs[label]
          df = df.query(mask)
          df = df.sample(n=Ntrain, random_state = seed)
          self.dfs[label].loc[ df.index.isin(df.index) ,'Train'] = 1 

      def AddNewColumnToAll(self,column,f):
          """ Adds a new column based on a function which takes dataframes as
              an  argument """
          for key, value in self.dfs.items():
               self.dfs[key][column] = f(value)

      def AddNewColumnToX(self,column,f,label):
          """ Adds a new column based on a function which takes dataframes as
              an  argument """
          self.dfs[label][column] = f(self.dfs[label])
          
      def GetDataframe(self,label):
          return(self.dfs[label])



