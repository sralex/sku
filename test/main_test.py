from unittest import TestCase
from sku import NKBinsDiscretizer
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def get_bins_sku(data,n_bins,strategy):
    nan_transformer = NKBinsDiscretizer(n_bins = n_bins,strategy=strategy)
    nan_transformer.fit(data)
    return nan_transformer.bins_[0] #np.alltrue(np.round(bins_sku_uniform,2) == np.round(bins_sklearn_uniform,2))

def get_bins_sklearn(data,n_bins,strategy):
    
    sklearn_transformer = KBinsDiscretizer(n_bins = n_bins,encode="ordinal", strategy=strategy)
    sklearn_transformer.fit(data)
    
    return sklearn_transformer.bin_edges_[0]

def get_num_names_sku(data,n_bins,strategy):

    nan_transformer = NKBinsDiscretizer(n_bins = n_bins,strategy=strategy)
    nan_transformer.fit(data)

    return nan_transformer.names_[0].shape[0] #np.alltrue(np.round(bins_sku_uniform,2) == np.round(bins_sklearn_uniform,2))

def get_num_names_sklearn(data,n_bins,strategy):
    
    sklearn_transformer = KBinsDiscretizer(n_bins = n_bins,encode="ordinal", strategy=strategy)
    sklearn_transformer.fit(data)
    
    return sklearn_transformer.n_bins_[0]

def get_data_sku(data,n_bins,strategy):
    nan_transformer = NKBinsDiscretizer(n_bins = n_bins,strategy=strategy)
    nan_transformer.fit(data)
    nan_result_uniform = nan_transformer.transform(data)
    unique_nan, counts_nan = np.unique(nan_result_uniform, return_counts=True)

    return np.sort(counts_nan)

def get_data_sklearn(data,n_bins,strategy):
    sklearn_transformer = KBinsDiscretizer(n_bins = n_bins,encode="ordinal", strategy=strategy)
    sklearn_transformer.fit(data)
    sklearn_result_uniform = sklearn_transformer.transform(data)
    unique_sklearn, counts_sklearn = np.unique(sklearn_result_uniform, return_counts=True)

    return np.sort(counts_sklearn)


class TestNKBinsDiscretizer(TestCase):
    def setUp(self):
        np.random.seed(42)
        self.data = np.random.random((100,1))
        
        self.data2 =np.concatenate([np.full((100,1),1),np.full((100,1),30)],axis=0)

        data_nan = self.data.copy()
        data_nan[20:30] = np.nan

        self.data_nan = data_nan
        self.data_not_nan = data_nan[~np.isnan(data_nan)].reshape((-1,1))

    def test_num_of_bins(self):

        transformer = NKBinsDiscretizer(n_bins = 5)
        transformer.fit(self.data)
        transformer.transform(self.data)
        self.assertEqual(transformer.bins_[0].shape[0],6)

    def test_num_bins_uniform(self):

        test1 = self.data,10,"uniform"
        test2 = self.data,50,"uniform"
        test3 = self.data,100,"uniform"
        test4 = self.data2,10,"uniform"
        test5 = self.data2,50,"uniform"
        test6 = self.data2,100,"uniform"

        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test1),2) == np.round(get_bins_sku(*test1),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test2),2) == np.round(get_bins_sku(*test2),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test3),2) == np.round(get_bins_sku(*test3),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test4),2) == np.round(get_bins_sku(*test4),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test5),2) == np.round(get_bins_sku(*test5),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test6),2) == np.round(get_bins_sku(*test6),2)))

    def test_num_bins_quantile(self):

        test1 = self.data,10,"quantile"
        test2 = self.data,50,"quantile"
        test3 = self.data,100,"quantile"
        test4 = self.data2,10,"quantile"
        test5 = self.data2,50,"quantile"
        test6 = self.data2,100,"quantile"


        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test1),2) == np.round(get_bins_sku(*test1),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test2),2) == np.round(get_bins_sku(*test2),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test3),2) == np.round(get_bins_sku(*test3),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test4),2) == np.round(get_bins_sku(*test4),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test5),2) == np.round(get_bins_sku(*test5),2)))
        self.assertTrue(np.alltrue(np.round(get_bins_sklearn(*test6),2) == np.round(get_bins_sku(*test6),2)))

    def test_data_transform_uniform(self):
        test1 = self.data,10,"uniform"
        test2 = self.data,50,"uniform"
        test3 = self.data,100,"uniform"
        test4 = self.data2,10,"uniform"
        test5 = self.data2,50,"uniform"
        test6 = self.data2,100,"uniform"


        self.assertTrue(np.alltrue(get_data_sklearn(*test1) == get_data_sku(*test1)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test2) == get_data_sku(*test2)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test3) == get_data_sku(*test3)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test4) == get_data_sku(*test4)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test5) == get_data_sku(*test5)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test6) == get_data_sku(*test6)))

    def test_data_transform_quantile(self):

        test1 = self.data,10,"quantile"
        test4 = self.data2,10,"quantile"
        test5 = self.data2,50,"quantile"
        test6 = self.data2,100,"quantile"

        self.assertTrue(np.alltrue(get_data_sklearn(*test1) == get_data_sku(*test1)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test4) == get_data_sku(*test4)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test5) == get_data_sku(*test5)))
        self.assertTrue(np.alltrue(get_data_sklearn(*test6) == get_data_sku(*test6)))

    def test_nan_values(self):

        test1_not_nan = self.data_not_nan,10,"uniform"
        test1_nan = self.data_nan,10,"uniform"

        test2_not_nan = self.data_not_nan,10,"quantile"
        test2_nan = self.data_nan,10,"quantile"

        self.assertTrue((get_num_names_sklearn(*test1_not_nan) + 1) == get_num_names_sku(*test1_nan))
        self.assertTrue((get_num_names_sklearn(*test2_not_nan) + 1) == get_num_names_sku(*test2_nan))
