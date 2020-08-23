from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectorMixin


# inspired in
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
                            features = [f'vec_{old_features[0]}_{f}' for f in step.get_feature_names()]
                        else:
                            print(step)
                            print(type(step))
                            features = step.get_feature_names(features)
                    elif isinstance(estimator, SelectorMixin):
                        features = np.array(features)[estimator.get_support()]
            else:
                step = estimator
                if hasattr(step, 'get_feature_names'):
                    if isinstance(step, CountVectorizer):
                        features = [f'vec_{f}' for f in step.get_feature_names()]
                    else:
                        features = step.get_feature_names(features)
                elif isinstance(estimator, SelectorMixin):
                    features = np.array(features)[estimator.get_support()]
        elif estimator == 'passthrough':
            features = feature_names_in[features]
        elif estimator == 'drop':
            features = []
        current_features = {f: None for f in features}
        output_features.update(current_features)
    return list(output_features.keys())
