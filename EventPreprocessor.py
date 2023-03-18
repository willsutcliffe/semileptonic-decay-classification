
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from root_pandas import read_root

class EventPreprocessor:
      """ Class which handles BDT preprocessing """

      def __init__(self, data = {}, tree ='',usereadroot=False):
          """ Constructor of BDTPreprocessor"""
          self.dfs = {}
          if usereadroot:
            for key, value in data.items():
               self.dfs[key] = read_root(value, tree)
          else:
            self.dfs = data
       
      def Combine(self,one,two,label,sample='Sample'):
          """ combines two data frames in the dictionary """
          self.dfs[one[0]][sample]=one[1]
          self.dfs[two[0]][sample]=two[1]
          self.dfs[label] = pd.concat([self.dfs[one[0]], self.dfs[two[0]]], ignore_index=True)
          self.dfs[label]['Index'] = self.dfs[label].index 
          del self.dfs[one[0]]
          del self.dfs[two[0]]



      def CombineMultiple(self,names,label):
          """ combines multiple dataframes to one and reorders according to
               an index """
          toadd = []
          for name in names:
              toadd.append(self.dfs[name])
          self.dfs[label] = pd.concat(toadd).sort_values(by=['Index'])

      def MultiplyWeights(self,inputweights,outputweightname):
          for key, value in self.dfs.items():
             self.dfs[key].loc[:,outputweightname] = 1
             for weight in inputweights:
                 self.dfs[key].loc[:,outputweightname] = self.dfs[key].loc[:,outputweightname]*self.dfs[key].loc[:,weight]

      def ApplyMask(self,mask):
          for key, value in self.dfs.items():
               self.dfs[key] = dfs[key].query(mask)
     
      def ApplyOneMask(self,mask,label, newlabel =''):
          if(len(newlabel)==0):
             self.dfs[label] = self.dfs[label].query(mask)
          else:
             self.dfs[newlabel] = self.dfs[label].query(mask)

      def FillNaNs(self,value,columns=None):
          """ Fill any NaNs in dataframe """
          for key, df in self.dfs.items():
              if columns != None:
                  self.dfs[key][columns] = self.dfs[key][columns].fillna(value)
              self.dfs[key] = self.dfs[key].fillna(value)

      def RandomiseTestTrain(self,mask,label,Ntrain,weightvar,seed=None):
          """ Adds a Train column initialised to 0 and sets 
              events within a given mask to having train = 1 """
          if('Train' not in self.dfs[label].columns):
            self.dfs[label].loc[:,'Train'] = 0 
            self.dfs[label].loc[:,'TestTrainWeight'] = 1
          df = self.dfs[label]
          df = df.query(mask)
          df = df.sample(n=Ntrain, random_state = seed)
          self.dfs[label].loc[ self.dfs[label].index.isin(df.index) ,'Train'] = 1 
          dftest =  self.dfs[label].loc[ ~self.dfs[label].index.isin(df.index)]
          dftest = dftest.query(mask)
          self.dfs[label].loc[ self.dfs[label].index.isin(dftest.index) ,'TestTrainWeight'] = np.sum(self.dfs[label].query(mask)[weightvar])/np.sum(self.dfs[label].query(mask).query('Train == 0')[weightvar])

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


      def WriteFile(self,label,filename = 'output.root'):
          self.dfs[label].to_root(filename,key='tree')
