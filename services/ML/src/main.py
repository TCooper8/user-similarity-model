import os
import logging
import warnings

from analyzer.training import train_user_similarities

from flask import Flask
from flask import jsonify

warnings.simplefilter(action='ignore')
logging.getLogger().setLevel(logging.INFO)

def main():
  model = train_user_similarities()

  app = Flask(__name__)

  @app.route('/health')
  def hello():
    return "Service is running", 200

  @app.route('/users/<user_handle>/similar')
  def user_similar(user_handle):
    try:
      user_handle = float(user_handle)
    except Exception as e:
      return ("%s" % e), 400

    logging.info("Getting users similar to %s" % user_handle)
    similar = model.similar_to(user_handle)
    return jsonify(list(similar))

  app.run(host='0.0.0.0', port=int(os.environ.get("PORT")))

if __name__ == '__main__':
  main()