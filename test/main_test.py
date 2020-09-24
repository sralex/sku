from unittest import TestCase
from sku import NKBinsDiscretizer
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def create_sklearn_kb_bins(data,n_bins,strategy):

    nan_transformer = NKBinsDiscretizer(n_bins = n_bins,strategy=strategy)
    nan_transformer.fit(data)
    bins_sku_uniform = nan_transformer.bins[0]

    sklearn_transformer = KBinsDiscretizer(n_bins = n_bins,encode="ordinal", strategy=strategy)
    sklearn_transformer.fit(data)
    
    bins_sklearn_uniform = sklearn_transformer.bin_edges_[0]

    return np.alltrue(np.round(bins_sku_uniform,2) == np.round(bins_sklearn_uniform,2))


def create_sklearn_kb_data(data,n_bins,strategy):

    nan_transformer = NKBinsDiscretizer(n_bins = n_bins,strategy=strategy)
    nan_transformer.fit(data)

    sklearn_transformer = KBinsDiscretizer(n_bins = n_bins,encode="ordinal", strategy=strategy)
    sklearn_transformer.fit(data)
    


    nan_result_uniform = nan_transformer.transform(data)
    sklearn_result_uniform = sklearn_transformer.transform(data)

    unique_nan, counts_nan = np.unique(nan_result_uniform, return_counts=True)
    unique_sklearn, counts_sklearn = np.unique(sklearn_result_uniform, return_counts=True)

    counts_nan = np.sort(counts_nan)
    counts_sklearn = np.sort(counts_sklearn)
    print(counts_nan)
    print(counts_sklearn)
    return np.alltrue(counts_nan == counts_sklearn)

class TestNKBinsDiscretizer(TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.random((100,1))
        self.data2 =np.concatenate([np.full((100,1),1),np.full((100,1),30)],axis=0)

    def test_num_of_bins(self):
        
        transformer = NKBinsDiscretizer(n_bins = 5)
        transformer.fit(self.data)
        transformer.transform(self.data)
        self.assertEqual(transformer.bins[0].shape[0],6)

    def test_num_bins_uniform(self):

        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data,10,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data,50,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data,100,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data2,10,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data2,50,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data2,100,"uniform")))

    def test_num_bins_quantile(self):

        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data,10,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data,50,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data,100,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data2,10,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data2,50,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_bins(self.data2,100,"quantile")))

    def test_data_transform_uniform(self):
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data,10,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data,10,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data,50,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data,100,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data2,10,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data2,10,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data2,50,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data2,50,"quantile")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data2,100,"uniform")))
        self.assertTrue(np.alltrue(create_sklearn_kb_data(self.data2,100,"quantile")))