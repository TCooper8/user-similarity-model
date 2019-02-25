import pandas

from sklearn.metrics.pairwise import cosine_similarity

# Method not being used currently.
def calc_similarity(frame, col_a, col_b):
  """
    This method is meant to generate a similarity matrix from two columns.
  """

  mat = pandas.crosstab(frame[col_a], frame[col_b])

  similarity = cosine_similarity(mat)
  return pandas.DataFrame(
    data = similarity,
    index = mat.index,
    columns = mat.index,
  )