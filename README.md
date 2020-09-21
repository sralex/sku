# sku
This is a small exercise to practice the python setuptools, here you will find a small library where I will be adding auxiliary functions for scikit-learn. For now it only has the function get_feature_names, which receives a Transformer with some pipelines and returns the columns of the data, example:
```
...
transformer = Pipeline (steps = [
     ...
     ('onehot', OneHotEncoder (handle_unknown = 'ignore'))
     ...
])
...

column_trans = ColumnTransformer (
     transformers = [
           ...
           ('cat', transformer, categorical_features)
           ...
           ],
         remainder = "drop")

column_trans.fit (X)

columns = list (get_feature_names (column_trans))
print (columns)

["cat_0","cat_1",...]

```
