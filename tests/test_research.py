"""Small behavioral checks, independent of historical market data."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from lrm.core import execute, metrics, hamilton_em, _filter_smooth
from lrm.experiments import apply_hmm


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.idx=pd.date_range('2020-01-01',periods=4)
        self.returns=pd.DataFrame({'STOCK':[0.,.2,.1,0.], 'CASH':0.},index=self.idx)
        self.signal=pd.DataFrame([[1.,0.]],index=self.idx[:1],columns=self.returns.columns)

    def test_next_close_timing_and_self_financing_fee(self):
        result=execute(self.signal,self.returns,return_details=True)
        np.testing.assert_allclose(result['net'],[0,1/1.0007-1,.1,0],atol=1e-12)
        np.testing.assert_allclose(result['cost'],.0007*result['turnover'],atol=1e-12)
        np.testing.assert_allclose(result['weights'].sum(axis=1),1)

    def test_cash_does_not_trade_or_pay_fees(self):
        self.signal.iloc[0]=[0,1]
        result=execute(self.signal,self.returns,return_details=True)
        np.testing.assert_allclose(result['net'],0)
        np.testing.assert_allclose(result['turnover'],0)

    def test_passive_drift_is_not_turnover(self):
        self.signal.iloc[0]=[.5,.5]
        result=execute(self.signal,self.returns,return_details=True)
        self.assertEqual(result['turnover'].iloc[2],0)
        self.assertGreater(result['weights']['STOCK'].iloc[3],.5)

    def test_unquoted_trade_is_rejected(self):
        quotes=pd.DataFrame(True,index=self.idx,columns=self.returns.columns)
        quotes.loc[self.idx[1],'STOCK']=False
        with self.assertRaises(ValueError): execute(self.signal,self.returns,tradable=quotes)

    def test_missing_held_return_is_rejected(self):
        self.returns.loc[self.idx[2],'STOCK']=np.nan
        with self.assertRaises(ValueError): execute(self.signal,self.returns)

    def test_negative_weights_are_rejected(self):
        self.signal.iloc[0]=[1.2,-.2]
        with self.assertRaises(ValueError): execute(self.signal,self.returns)

    def test_drawdown_includes_initial_capital(self):
        self.assertAlmostEqual(metrics(pd.Series([-.2,0.]))['maxdd'],-.2)

    def test_selection_ablation_preserves_hmm_exposure(self):
        pure=pd.DataFrame([[.5,.5,0.,0.]],columns=['A','B','CASH','GLD'])
        hmm=pd.DataFrame([[.1,.2,.35,.35]],columns=pure.columns)
        new=apply_hmm(pure,hmm,['A','B'])
        np.testing.assert_allclose(new.iloc[0],[.15,.15,.35,.35])


class HMMTests(unittest.TestCase):
    def test_student_t_filter_matches_final_parameters(self):
        rng=np.random.default_rng(7)
        state=np.repeat([0,1,0,1,0],120)
        y=np.array([.05,-.1])[state]+np.array([.8,2.5])[state]*rng.standard_t(8,len(state))
        out=hamilton_em(y,n_iter=300,dist='t')
        self.assertTrue(out['converged'])
        filt,_,_,ll,ok=_filter_smooth(y,out['mu'],out['sig2'],out['P'],out['nu'],'t',1e-8,out['pi'])
        self.assertTrue(ok)
        np.testing.assert_allclose(filt,out['filt'],atol=1e-10,rtol=0)
        np.testing.assert_allclose(filt.sum(axis=1),1)
        self.assertAlmostEqual(ll,out['loglik'],places=8)

    def test_first_observation_uses_pi_without_transition(self):
        y=np.array([0.,.1]);mu=np.array([0.,0.]);sig2=np.array([1.,1.])
        pi=np.array([.9,.1]);P=np.array([[.1,.9],[.8,.2]])
        filt,_,_,_,ok=_filter_smooth(y,mu,sig2,P,8,'t',1e-8,pi)
        self.assertTrue(ok)
        np.testing.assert_allclose(filt[0],pi)


if __name__=='__main__': unittest.main()
