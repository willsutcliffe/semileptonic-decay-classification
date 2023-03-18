from BDT.Trainer import Trainer
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import plot_importance

class XGBoostTrainer(Trainer):
      """ Class wrapping xgboost """

      def __init__(self, _data,_hyperpars,_features,_weightvar,_targetlabel='Target',_trainlabel='Train', _outputlabel='BDT_prediction'):
          """ Constructor of XGBoostTrainer"""
          super().__init__(_data)
          self.hyperpars = _hyperpars
          self.features = _features
          self.weightvar= _weightvar
          self.trainlabel= _trainlabel
          self.targetlabel= _targetlabel
          self.outputlabel = _outputlabel
          self.PrepareData()
          self.num_round = 400

      def PrepareData(self):
          self.traindata = self.data.query('{} == 1'.format(self.trainlabel))
          self.dtrain = xgb.DMatrix( self.traindata[self.features].values, label = self.traindata[self.targetlabel], weight = self.traindata[self.weightvar])
          self.testdata = self.data.query('{} == 0'.format(self.trainlabel))
          self.dtest = xgb.DMatrix( self.testdata[self.features].values, label = self.testdata[self.targetlabel], weight = self.testdata[self.weightvar])
          self.ddata = xgb.DMatrix( self.data[self.features].values, label = self.testdata[self.targetlabel], weight = self.testdata[self.weightvar])
      

      def setHyperpars(self,pars):
          """" Sets hyperparamters
               @param pars"""
          self.hyperpars = pars
          
      def train(self):
         """ Trains the xgboost classifer """
         self.classifier = xgb.train(self.hyperpars, self.dtrain, self.num_round)

      def predict(self):
         """ Inter trained classifier on test and train data """
         self.testdata[self.outputlabel] =  self.classifier.predict(self.dtest)
         self.traindata[self.outputlabel] = self.classifier.predict(self.dtrain)
         self.data[self.outputlabel] = self.classifier.predict(self.ddata)
      
      def GetTestData(self):
         return(self.testdata)

      def GetTrainData(self):
         return(self.traindata)

      def GetSignalData(self):
         return(self.data.query('{} == 1'.format(self.targetlabel)))

      def GetBackgroundData(self):
         return(self.data.query('{} == 0'.format(self.targetlabel)))

      def savemodel(self,name):
         """ Save xgboost module
              @param name name of saved model"""
         self.classifier.save_model(name)

      def savetraindata(self,name):
         """ Save xgboost module
              @param name name of saved train data file"""
         self.traindata['Target']=  self.traindata['Target'].astype(np.float32)
         self.traindata['TotalWeight']=  self.traindata['TotalWeight'].astype(np.float32)
         self.traindata.to_root(name,key='tree')

      def savetestdata(self,name):
         """ Save xgboost module
              @param name name of saved train data file"""
         self.testdata['Target']=  self.testdata['Target'].astype(np.float32)
         self.testdata['TotalWeight']=  self.testdata['TotalWeight'].astype(np.float32)
         self.testdata.to_root(name,key='tree')
 
      def PlotClassifiers(self,range_BDT = (0,1),bins=10,output='classifier_test_train_comparison',textlabel='', train_cut=None, test_cut=None):
         """ Abstract base method which carries out the testing """
         sig_test = self.testdata.query("{} == 1".format(self.targetlabel))
         bkg_test = self.testdata.query("{} == 0".format(self.targetlabel))
         if test_cut != None:
           sig_test = sig_test.query(test_cut)
           bkg_test = bkg_test.query(test_cut)
         sig_train = self.traindata.query("{} == 1".format(self.targetlabel))
         bkg_train = self.traindata.query("{} == 0".format(self.targetlabel))
         if train_cut != None:
           sig_train = sig_train.query(train_cut)
           bkg_train = bkg_train.query(train_cut)
         data = {'Signal test' : sig_test, 'Background test': bkg_test, 'Signal train' : sig_train,
                 'Background train' : bkg_train}
         colors = {'Signal test' : 'lightskyblue', 'Background test':'indianred', 'Signal train' : 'blue',
                 'Background train' : 'red'}
         for key,data in data.items(): 
              y, bin_edges = np.histogram(data[self.outputlabel],bins=bins, range=range_BDT,weights=data[self.weightvar])
              y2, bin_edges = np.histogram(data[self.outputlabel],bins=bins, range=range_BDT,weights=data[self.weightvar]**2)
              bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
              plt.errorbar(bin_centers,y/np.sum(y),yerr = np.sqrt(y2)/np.sum(y),markersize=0.2,fmt='s',color=colors[key])
              plt.step(bin_edges, np.append([0],y)/np.sum(y),where="pre",color=colors[key],label=key)

         plt.xlim(range_BDT)
         plt.ylim(bottom=0,top=0.14)
         plt.figtext(0.15,0.85,textlabel,fontdict={'size':14})
         plt.xlabel("BDT output")
         #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
         plt.legend(loc='upper center', prop={'size':12})
         plt.tight_layout()
         plt.savefig('{}.pdf'.format(output))
         plt.savefig('{}.png'.format(output))
         plt.show()
         plt.close()

        
         
      def test(self):
         """ Abstract base method which carries out the testing """


      def ROC(self):
         """ Abstract base method which generates a ROC curve fo
             for testing and training  """


      def efficiency(self,df, cut):
         n_tot = np.sum(df[self.weightvar])
         n_sig = np.sum(df.query('{} > {}'.format(self.outputlabel, cut))[self.weightvar])
         return(n_sig /n_tot)


      def make_roc_curve(self,points, df_sig, df_bkg):
         datax = []
         datay = []
         for point in points:
              fsig = self.efficiency(df_sig, point)
              fbkg = 1 - self.efficiency(df_bkg, point)
              datax.append(fsig)
              datay.append(fbkg)
         return datax, datay

      def plotROCcurve(self, points=np.linspace(0, 1, num=51),output='ROC_curve',textlabel='', train_cut=None, test_cut=None):
          testdata = self.testdata
          traindata = self.traindata
          if train_cut != None:
            traindata = traindata.query(train_cut)
          if test_cut != None:
            testdata = testdata.query(test_cut)

          x_test, y_test = self.make_roc_curve(points, testdata.query("{} == 1".format(self.targetlabel)),
                                          testdata.query("{} == 0".format(self.targetlabel)))
          x_train, y_train = self.make_roc_curve(points, traindata.query("{} == 1".format(self.targetlabel)),
                                            traindata.query("{} == 0".format(self.targetlabel)))
          print(x_test, y_test)
          fig, ax = plt.subplots()
          ax.scatter(x_test, y_test, c='orange', label='test', edgecolors='none')
          ax.scatter(x_train, y_train, c='green', label='train', edgecolors='none')
          ax.legend()
          ax.grid(True)
          plt.xlabel("Signal reconstruction eff.")
          plt.ylabel("Bkg rejection eff.")
          plt.figtext(0.2,0.4,textlabel,fontdict={'size':14})
          plt.savefig('{}.pdf'.format(output))
          plt.savefig('{}.png'.format(output))
          plt.show()
          plt.close()

      def plotVariablesByImportance(self,output='feature_importance',textlabel=''):
          plot_importance(self.classifier)
          plt.figtext(0.8,0.3,textlabel,fontdict={'size':14})
          plt.savefig('{}.pdf'.format(output))
          plt.savefig('{}.png'.format(output))
          plt.show()
          plt.close()

      def AROC(self):
          """
          Calculates the area under the receiver oeprating characteristic curve (AUC ROC)
          """
          return self.area_ROC(self.testdata['BDT_prediction'].values, self.testdata[self.targetlabel].values)


      def area_ROC(self, p, t):
           """
           Calculates the area under the receiver oeprating characteristic curve (AUC ROC)
           @param p np.array filled with the probability output of a classifier
           @param t np.array filled with the target (0 or 1)
           """
           N = len(t)
           T = np.sum(t)
           index = np.argsort(p)
           efficiency = (T - np.cumsum(t[index])) / float(T)
           purity = (T - np.cumsum(t[index])) / (N - np.cumsum(np.ones(N)))
           purity = np.where(np.isnan(purity), 0, purity)
           return np.abs(np.trapz(purity, efficiency))
