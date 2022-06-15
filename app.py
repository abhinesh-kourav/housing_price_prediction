from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return 'Learned how to create a CI CD pipeline!!!'

if __name__ == '__main__':
    app.run(debug=True)