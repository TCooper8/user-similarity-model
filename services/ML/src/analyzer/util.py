import numpy

def drop_sparse_columns(frame):
  return frame[frame.columns[frame.notnull().mean() > 0.5]]

def drop_high_cardinality(frame, max=100, max_percent=0.9):
  return frame.drop(
    [ col for col, val in frame.nunique().iteritems()
        if (frame[col].dtype == numpy.dtype('object'))
          and (val > max or val >= (frame.shape[0] * max_percent))
    ],
    axis=1,
  )