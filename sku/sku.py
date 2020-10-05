from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import  check_is_fitted, FLOAT_DTYPES, check_array, _deprecate_positional_args
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# inspired on
# https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
def get_feature_names(transformer):
    output_features = {}
    for name, estimator, features in transformer.transformers_:
        if name != "remainder":
            if isinstance(estimator, Pipeline):
                for step in estimator:
                    
                    if hasattr(step, 'get_feature_names'):
                        for feature in features:
                            try:
                                del output_features[feature]
                            except KeyError:
                                pass
                        if isinstance(step, CountVectorizer):
                            old_features = features
                            features = ['vec_{}_{}'.format(old_features[0],f) for f in step.get_feature_names()]
                        else:
                            #print(step)
                            #print(type(step))
                            features = step.get_feature_names(features)
                    else:
                        if isinstance(estimator, SelectorMixin):
                            features = np.array(features)[estimator.get_support()]
                        else:
                            if hasattr(step, 'bin_edges_'):
                                old_features = features
                                features = []
                                for i in range(len(old_features)):
                                    features += ['KBin_{}_{}'.format(f,old_features[i]) for f in step.bin_edges_[i][0:-1]]
                       
            else:
                step = estimator
                if hasattr(step, 'get_feature_names'):
                    if isinstance(step, CountVectorizer):
                        features = ['vec_{}'.format(f) for f in step.get_feature_names()]
                    else:
                        features = step.get_feature_names(features)
                else:
                    if isinstance(step, SelectorMixin):
                        features = np.array(features)[estimator.get_support()]
                    else:
                        if hasattr(step, 'bin_edges_'):
                            old_features = features
                            features = []
                            for i in range(len(old_features)):
                                features += ['KBin_{}_{}'.format(f,old_features[i]) for f in step.bin_edges_[i][0:-1]]
        elif estimator == 'drop':
            features = []
        current_features = {f: None for f in features}
        output_features.update(current_features)
    return list(output_features.keys())

# similar to KBinsDiscretizer but with nan support 
#https://scikit-learn.org/stable/developers/develop.html
class NKBinsDiscretizer(BaseEstimator,TransformerMixin):

    @_deprecate_positional_args
    def __init__(self, n_bins = 5, strategy = "uniform", label_mode = "range"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.label_mode = label_mode

       
    def fit(self, X, y=None, *args, **kwargs):

        valid_label_mode = ('range', 'ordinal')
        if self.label_mode not in valid_label_mode:
            raise ValueError("Valid options for 'encode' are {}. "
                             "Got encode={!r} instead."
                             .format(valid_label_mode, self.label_mode))


        valid_strategy = ('uniform', 'quantile')
        if self.strategy not in valid_strategy:
            raise ValueError("Valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, self.strategy))


        X = self._validate_data(X, dtype='numeric', force_all_finite=False)

        self.bins_ = {}
        self.names_ = {}

        n_columns = X.shape[1]

        self.n_bins_ = self._validate_bins(self.n_bins,n_columns)

        for i in range(n_columns):
            column = X[:,i]
            is_nan = np.isnan(column)
            not_nan_array = column[~is_nan]


            n_bins = self.n_bins_[i]

            if self.strategy == "uniform":
                self.bins_[i] = np.linspace(not_nan_array.min(), not_nan_array.max(), n_bins) if type(n_bins) != list else n_bins
            else:
                # TODO: catch error when n_bins is a list
                self.bins_[i] = np.unique(np.percentile(not_nan_array,np.linspace(0, 100, n_bins)))
                
            self.names_[i] = ["[{} - {})".format(a,b) for a,b in zip(self.bins_[i][:-1],self.bins_[i][1:])]

            self.names_[i] = np.array(self.names_[i] + ["nan"] if is_nan.sum() > 0 else self.names_[i])
        return self
   

    def _validate_bins(self,bins,n_columns):
        if type(bins) != list:
            bins = [bins + 1] * n_columns
        else:
            if len(bins) != n_columns:
                raise ValueError("Bins length {} is different to columns length ({})."
                                 .format(len(bins), n_columns))
        return bins


    def transform(self, X, y=None, *args, **kwargs):
        
        check_is_fitted(self)
        
        #X = self._validate_data(X, dtype='numeric', force_all_finite=False)
        
        Xt = check_array(X, copy=True, dtype=object, force_all_finite=False)

        for i in range(Xt.shape[1]):
            column = Xt[:,i]
            bins = self.bins_[i].copy()
            bins[0] -= 0.001
            digitized = np.digitize(column, bins, right = True)
            if not self.label_mode == "ordinal":
                digitized = self.names_[i][digitized -1]
            Xt[:,i] = digitized
        
        return Xt