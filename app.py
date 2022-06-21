from flask import Flask
from housing.logger import logging
from housing.exception import HousingException
import sys

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    try:
        raise Exception('We are testing custom exception.')
    except Exception as e:
        housing = HousingException(e, sys)
        logging.info(housing.error)

    logging.info('We are testing logging module')
    return 'Learned how to create a CI CD pipeline!!!'

if __name__ == '__main__':
    app.run(debug=True)