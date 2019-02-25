import numpy

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

# This class is not currently being used.
# This is here mainly to show that categoricals should not simply be enumerated.
# You can get better training results from impact coding your categoricals.
class ImpactCoder(TransformerMixin):
  def __init__(self, columns):
    if type(columns) == str:
      columns = [columns]
    self.columns = columns
    self.y_bar = 0.0
    self.dicts = dict()

  def fit(self, X, y=None):
    for col in self.columns:
      d = dict()
      if col in X.columns:
        for cat in X[col].unique():
          sub = X[X[col] == cat]
          i = sub.index
          d[cat] = X[col].mean()
        self.dicts[col] = d
    return self

  def transform(self, X, y=None):
    for col in self.columns:
      if col in X.columns:
        d = self.dicts[col]
        X[col] = numpy.where(
          X[col].isin(d.keys()),
          X[col].replace(d),
          self.y_bar,
        ) - self.y_bar

        X[col] = pandas.to_numeric(X[col], errors='coerce')

    return X

class CategoryEncoder(TransformerMixin):
  """
    This class is meant to be part of a pipeline.
    This will encode categoricals.
  """

  def __init__(self):
    self.__cats = []
    self.__enc = dict()

  def columns(self):
    return self.__cats

  def fit(self, X, Y=None):
    self.__cats = []
    
    for col in X.columns:
      if X[col].dtype == numpy.dtype('object'):
        self.__cats.append(col)
        self.__enc[col] = LabelEncoder().fit(X[col].astype(str))

    return self

  def transform(self, X, Y=None):
    for col in self.__cats:
      if col in X.columns:
        enc = self.__enc[col]
        d = dict(zip(enc.classes_, enc.transform(enc.classes_)))
        X[col] = X[col].map(d).fillna(0.0)

    return X

class FramePrep(TransformerMixin):
  """
    This class is meant to be part of a pipeline.
    This will encode categoricals and normalize numerics.
  """

  def __init__(self):
    self.__enc = CategoryEncoder()
    self.__scaler = StandardScaler()

  def fit(self, X, Y=None):
    # Fit the categoricals to the encoder.
    self.__enc = self.__enc.fit(X, Y)

    # Now fit the numerics to the scaler.
    # Get the columns from the encoder, this is so we can diff the numeric columns.
    cats = self.__enc.columns()
    nums = X.loc[:, ~X.columns.isin(cats)].columns

    # NOTE: In certain cases, you should only be normalizing numeric floats.
    # Some cases you will have non-numeric integer columns (enums for example)
    self.__scaler = self.__scaler.fit(X[nums].astype(float))

    return self

  def transform(self, X, Y=None):
    cats = self.__enc.columns()
    X = self.__enc.transform(X)
    nums = X.loc[:, ~X.columns.isin(cats)].columns

    # NOTE: In certain cases, you should only be normalizing numeric floats.
    # Some cases you will have non-numeric integer columns (enums for example)
    X[nums] = self.__scaler.transform(X[nums].astype(float))

    return X