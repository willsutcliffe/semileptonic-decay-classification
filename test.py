
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from BDT.EventPreprocessor import EventPreprocessor
from BDT.XGBoostTrainer import XGBoostTrainer

def BDT_get_eff_hybrid(df,cut,wcol,cutcol):
    
    n_tot = np.sum(df[wcol])
    n_sig = np.sum(  df.query('{} > {}'.format(cutcol,cut))[wcol]  )
    eff = n_sig/(n_tot)
    return eff

def BDT_get_eff_hybrid2(df, BDT_cut):
    
    n_tmp, bins_tmp, _  = plt.hist(df['BDT_prediction'], 
            bins= (0, BDT_cut ,1), 
            weights= df['_hybridweight_'])
    plt.close() 
        
    n_bkg = n_tmp[0]
    n_sig = n_tmp[1]                                        
                                       
    eff = n_sig/(n_sig + n_bkg)
    
    return eff

data = { 'ulnu_ch' :'/storage/9/belle/analysis-inclusive-b2ulnu/20180821/output_hybrid/output_mc_ulnu_hybrid_charged_20180821_preselect.root',
         'ulnu_md' : '/storage/9/belle/analysis-inclusive-b2ulnu/20180821/output_hybrid/output_mc_ulnu_hybrid_mixed_20180821_preselect.root',
         'clnu_ch' : '/storage/9/belle/analysis-inclusive-b2ulnu/20180808/pre_selected/generic/mixed/bingen_md_s0_preselect_20180808.root',
         'clnu_md' : '/storage/9/belle/analysis-inclusive-b2ulnu/20180808/pre_selected/generic/mixed/bingen_md_s0_preselect_20180808.root' }

Ntrain = 30000

# Do common event processing on datasets 
p = EventPreprocessor(data,'tree')
p.Combine(('ulnu_ch','ch'),('ulnu_md','md'),'ulnu','BMeson')
p.Combine(('clnu_ch','ch'),('clnu_md','md'),'clnu','BMeson')
p.FillNaNs(10000)
p.AddNewColumnToAll('vertexfit', lambda df: np.where((df['gx_vtx_chi2']>0)&(df['gx_vtx_dgf']>0), np.log10(df.gx_vtx_chi2/df.gx_vtx_dgf), -100))
p.AddNewColumnToX('Target', lambda df: 1,'ulnu')
p.AddNewColumnToX('Target', lambda df: 0,'clnu')
p.AddNewColumnToX('_hybridweight_', lambda df: 1,'clnu')
p.RandomiseTestTrain('((btag_m_bc >= 5.27) & (abs(btag_pcode_b) == 511))','ulnu',Ntrain)
p.RandomiseTestTrain('((btag_m_bc >= 5.27) & (abs(btag_pcode_b) != 511))','ulnu',Ntrain)
p.RandomiseTestTrain('((btag_m_bc >= 5.27) & (abs(btag_pcode_b) == 511))','clnu',Ntrain)
p.RandomiseTestTrain('((btag_m_bc >= 5.27) & (abs(btag_pcode_b) != 511))','clnu',Ntrain)

# Retrieve and finalise samples for training
ulnu = p.GetDataframe('ulnu')
clnu = p.GetDataframe('clnu')
data  = ulnu.append(clnu)
B0_data = data.query('((btag_m_bc >= 5.27) & (abs(btag_pcode_b) == 511))')
BP_data = data.query('((btag_m_bc >= 5.27) & (abs(btag_pcode_b) != 511))')

# Set up and train BDT
features = ['gmiss_m2','event_nlep', 'event_q','event_nk', 'event_nks', 'veto_slowCharPi_missM2', 'veto_slowNeuPi_missM2', 'vertexfit', 'veto_slowNeuPi_cos_ThetaC', 'veto_slowCharPi_cos_ThetaC', 'veto_slowCharPi_cos_BY', 'veto_slowNeuPi_cos_BY']
hyperpars = {'max_depth':2, 'eta':0.1, 'silent':1, 'objective':'binary:logistic'}
B0_trainer = XGBoostTrainer(B0_data,hyperpars, features,'_hybridweight_')
B0_trainer.train()
B0_trainer.predict()
B0_trainer.savemodel('B0_BDT')
BP_trainer = XGBoostTrainer(BP_data,hyperpars, features,'_hybridweight_')
BP_trainer.train()
BP_trainer.predict()

#testsample = B0_trainer.GetTestData().append(BP_trainer.GetTestData())
#ulnutest = testsample.query("Target == 1")
#clnutest = testsample.query("Target == 0")
#fsig = BDT_get_eff_hybrid(ulnutest, 0.835,'_hybridweight_','BDT_prediction')
#fsig = BDT_get_eff_hybrid2(ulnutest, 0.835)
#fbkg = BDT_get_eff_hybrid2(clnutest, 0.835)


