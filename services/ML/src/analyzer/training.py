import pandas
import logging

from .prep import FramePrep
from .data import load_data
from .util import drop_high_cardinality, drop_sparse_columns
from .prep import CategoryEncoder

from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.base import TransformerMixin

def load(name, index=None):
  logging.info("Loading dataset %s", name)
  return load_data(name, index=index)

def list_similar(dataset, user_handle, pipeline, model, drop=[]):
  """
    This method is meant to be used by the SimilarityModel class as a utility.
    This method will use the given model to lookup which users are similar to the inpu user_handle.
  """
  # Get the rows which contain the user handle.
  indices = dataset.loc[dataset['user_handle'] == user_handle].index

  X = dataset
  # Drop user_handle
  X = X.drop(columns=drop)
  X = drop_high_cardinality(X, max=1000)
  X = drop_sparse_columns(X)
  # Run the dataset through the pipeline so that we can make row level predictions.
  X = pipeline.transform(X)

  # Make predictions.
  similar = set()
  for i in indices:
    if i >= X.shape[0]:
      continue
    neighbors = model.radius_neighbors([X.iloc[i]], return_distance=True)
    distances = neighbors[0][0]
    neighbor_indices = neighbors[1][0]

    for i, dist in zip(neighbor_indices, distances):
      row = dataset.iloc[[i]]
      similar.add(str(row.user_handle.values[0]))

  return similar

def prep(X, drop_columns=[]):
  """
    This function serves for simple preprocessing prep.
  """
  logging.info("Preparing frame %s", X.columns)

  if len(drop_columns) > 0:
    X = X.drop(columns=drop_columns)
  X = drop_high_cardinality(X, max=1000)
  X = drop_sparse_columns(X)
  return X

def create_pipeline(X):
  """
    This function will create a generic prep pipeline for reuse.
  """

  logging.info("Pretrain pipeline for frame %s", X.columns)
  # Create a pipeline.
  pipeline = make_pipeline(
    FramePrep(),
  ).fit(X)

  X = pipeline.transform(X)

  return pipeline, X
  
class SimilarityModel(object):
  """
    Class for training a user similarity model.
  """

  def __init__(self):
    self.__load_data = None
    self.__pipeline = None
    self.__model = None
    self.__drop_columns = None

  def train(self, load_data, radius=0.1, drop_columns=[]):
    """
      This method will train the model on the input data from the given load_data function.
      `radius` is referring to the distance between nodes in the similarity graph.
      `drop_columns` will drop the listed columns before training.
    """
    ## Initial data prep.
    X = load_data()
    logging.info("Training similarity model on frame %s", X.columns)

    X = prep(X, drop_columns=drop_columns)

    # Pretraining prep.
    pipeline, X = create_pipeline(X)

    # Model prep and training.
    model = NearestNeighbors(
      radius = radius,
    )
    model.fit(X)

    # Store results within class.
    self.__pipeline = pipeline
    self.__load_data = load_data
    self.__model = model
    self.__drop_columns = drop_columns

    return self

  def similar_to(self, user_handle):
    """
      Will return a set of user handles that are similar to the give user_handle.
    """
    data = self.__load_data()

    logging.info("Looking up similarity to %s from frame %s", user_handle, data.columns)

    return list_similar(
      data,
      user_handle,
      self.__pipeline,
      self.__model,
      drop = self.__drop_columns,
    )

class MultiSimilarityModel(object):
  """
    This class holds multiple similarity models, meant for model result aggregation.
  """

  def __init__(self, models):
    self.__models = models

  def similar_to(self, y):
    similar = set()
    for model in self.__models:
      similar.update(model.similar_to(y))

    return similar

def load_courses_data():
  """
    Loads courses_data from source.
    Unfortunately, some domain knowledge is required before training prep.
  """
  user_course_views = load('user_course_views.csv')
  course_tags = load('course_tags.csv')
  data = user_course_views.merge(
    course_tags,
    left_on='course_id',
    right_on='course_id',
    how = 'inner',
  )
  data.drop(columns='view_time_seconds')
  
  return data

def load_assessment_data():
  """
    Loads assessment data from source.
  """
  return load('user_assessment_scores.csv')

def load_interest_data():
  """
    Loads interest data from source.
  """
  return load("user_interests.csv")

def train_user_similarities(radius=0.01):
  """
    Utility for training a MultiSimilarityModel.
  """
  assessment_model = SimilarityModel().train(
    load_assessment_data,
    radius = radius,
    drop_columns = [
      'user_handle',
    ],
  )

  # Demonstrating a multi-model subsampling approach here.
  # Take samples, and train a model on each section of the data.
  # Accumulate predictions from each model.
  # Different approaches can be taken when subsampling the data, and when merging predictions.
  # Also, the models could be distributed to multiple nodes for training or for predictions.

  ## Note: This code is only here for review and to show that it can be done to gain efficiency.

  #interest_models = []
  #def load_interest_subsample (offset, limit):
  #  def f():
  #    frame = load_interest_data()
  #    logging.info("Reading subsample of range [%d:%d]" % (offset, limit))
  #    return frame.iloc[offset:limit]
  #  return f

  #for i in range(0, 30):
  #  n = 10000
  #  j = i
  #  model = SimilarityModel().train(
  #    load_interest_subsample(j * n, j * n + n - 1),
  #    radius = radius,
  #  )
  #  interest_models.append(model)
  #interest_model = MultiSimilarityModel(
  #  models = interest_models,
  #)
    
  interest_model = SimilarityModel().train(
    load_interest_data,
    radius = radius,
  )
  courses_model = SimilarityModel().train(
    load_courses_data,
    radius = radius,
    drop_columns=[
      'user_handle',
      'view_date',
    ],
  )

  # Join models together
  return MultiSimilarityModel(
    models = [
      assessment_model,
      interest_model,
      courses_model,
    ],
  )