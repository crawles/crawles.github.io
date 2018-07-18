---
layout: post
title: "How to normalize features in TensorFlow"
feature: "tf-feature-normalize.jpeg"
keywords: "Machine Learning. Data Science. Python. TensorFlow."
published: true
---

[Post here.](https://towardsdatascience.com/how-to-normalize-features-in-tensorflow-5b7b0e3a4177)

*TL;DR*
*When using tf.estimator, use the normalizer_fn argument in tf.feature_column.numeric_feature to normalize using the same parameters (mean, std, etc.) for training, evaluation, and serving.*

```python
def zscore(col):
  mean = 3.04
  std = 1.2
  return (col — mean)/std
feature_name = ‘total_bedrooms’
normalized_feature = tf.feature_column.numeric_column(
  feature_name,
  normalizer_fn=zscore)
```
