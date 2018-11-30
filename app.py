import uuid

from flask import Flask, request, abort, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = None

# def load_model():
#     global graph
#     graph = tf.get_default_graph()

@app.route('/linearregression', methods=['POST'])
def linearregression():
    if not request.json:
        abort(400)
    content = request.json

    if not content["dataUrl"]:
        abort(400)

    print("loading image from url:" + content['dataUrl'])

    df = pd.read_csv(content['dataUrl'])
    length = len(df.columns)
    x = df.iloc[:,0:length-2]
    y = df.iloc[:,length-1]


    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    clf = linear_model.LinearRegression()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(y_test[0:5])
    print(y_pred[0:5])
    print(r2_score(y_test,y_pred))

    plt.scatter(y_test, y_pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    filename = str(uuid.UUID) + ".png"
    plt.savefig("temp/" + filename)
    return send_file("temp/" + filename, mimetype='image/png')

if __name__ == '__main__':
    print(("* Loading Tensorflow model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(port=5002)