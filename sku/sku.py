from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectorMixin


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
