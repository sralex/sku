from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

# inspired on
# https://stackoverflow.com/questions/57528350/can-you-consistently-keep-track-of-column-labels-using-sklearns-transformer-api
def get_feature_names(transformer):
    feature_names_in = transformer._feature_names_in
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
        elif estimator == 'passthrough':
            features = feature_names_in[features]
        elif estimator == 'drop':
            features = []
        current_features = {f: None for f in features}
        output_features.update(current_features)
    return list(output_features.keys())

# similar to KBinsDiscretizer but with nan support 
class NKBinsDiscretizer(BaseEstimator):
    def __init__(self,n_bins = 5, strategy = "uniform"):
        self.bins = {}
        self.names = {}
        self._n_bins = n_bins + 1
        self._strategy = strategy
    
    def _iterator_pd_np(self, x):
        if isinstance(x,pd.core.frame.DataFrame):
            for column in x:
                yield x.loc[:,column]
        else:
            for column in x.T:
                yield column
            
    def fit(self, X, y=None, *args, **kwargs):
        for i,column in enumerate(self._iterator_pd_np(X)):
            not_nan_array = column[~np.isnan(column)]
            if self._strategy == "uniform":
                self.bins[i] = np.linspace(not_nan_array.min() - 0.0001, not_nan_array.max(), self._n_bins)
            else:
                self.bins[i] = np.unique(np.percentile(not_nan_array,np.linspace(0, 100, self._n_bins)))
                self.bins[i] -= 0.0001 
                
            self.names[i] = np.array(["[{} - {})".format(a,b) for a,b in zip(self.bins[i][:-1],self.bins[i][1:])]  + ["nan"])

    def transform(self, X, y=None, *args, **kwargs):
        X2 = []
        for i,column in enumerate(self._iterator_pd_np(X)):
            X2.append(self.names[i][np.digitize(column, self.bins[i], right = True) -1])
        
        X2 = np.array(X2).T
        if isinstance(X,pd.core.frame.DataFrame):
            X2 = pd.DataFrame(X2,columns=X.columns)
        return X2