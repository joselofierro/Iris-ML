# import pickle
import pandas
import psycopg2
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES

from form import TheForm
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

# Inicilizando la app Flask
app = Flask(__name__)
###################################
app.config['UPLOADED_PHOTOS_DEST'] = '/home/chuky97/PycharmProjects/DataScience/static/img'
photo = UploadSet('photos', IMAGES)
configure_uploads(app, photo)


####################################


# establecemos conexion con el Gestor de base de datos
def get_conexion():
    # establecemos la conexion a la BD
    conexion = psycopg2.connect(dbname="science-data", user="josefierro",
                                password="josefierro", host="localhost", port=5433)
    return conexion


@app.route('/build')
def build():
    # Load dataset url
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # leemos el archivo csv
    # dataset = pandas.read_csv(url, names=names)

    conexion = get_conexion()
    dataset = pandas.read_sql_query("SELECT sepal_length , sepal_width, petal_length, petal_width, clase  FROM iris",
                                    con=conexion)

    # Split-out validation dataset
    array = dataset.values
    # datos de entrenamiento (X -> V. Numericos, Y -> V. Nombres plantas )
    X = array[:, 0:4]
    Y = array[:, 4]

    validation_size = 0.20
    random_state = 7

    # entradas de pruebas y validacion
    X_train, x_validation, Y_train, y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=random_state)
    # algoritmo K-nearest a entrenar
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    predictions = knn.predict(x_validation)

    # dtos estadisticos
    print(accuracy_score(y_validation, predictions))
    confusion_matrix(y_validation, predictions)
    print(classification_report(y_validation, predictions))
    #

    # serializando a archivo en disco
    filename = '/home/chuky97/PycharmProjects/DataScience/model.pkl'
    joblib.dump(knn, filename)

    conexion.close()

    return "modelo creado"


@app.route('/', methods=['POST', 'GET'])
def clasificar(prediction=None):
    form = TheForm(request.form)
    if request.method == 'POST' and form.validate():
        sepal_length = form.param1.data
        sepal_width = form.param2.data
        petal_length = form.param3.data
        petal_width = form.param4.data

        flower_instance = [[sepal_length, sepal_width, petal_length, petal_width]]
        print(flower_instance)

        ml_model = joblib.load('/home/chuky97/PycharmProjects/DataScience/model.pkl')
        prediction = ml_model.predict(flower_instance)

    return render_template('home.html', form=form, prediction=prediction)


@app.route('/clasificar/<float:sl>/<float:sw>/<float:pl>/<float:pw>', methods=['GET'])
def clasificacion(sl, sw, pl, pw):
    if request.method == 'GET':
        sepal_length = sl
        sepal_width = sw
        petal_length = pl
        petal_width = pw

        caracteristicas = [[sepal_length, sepal_width, petal_length, petal_width]]
        ml_model = joblib.load('/home/chuky97/PycharmProjects/DataScience/model.pkl')
        prediction = ml_model.predict(caracteristicas)
        return jsonify({'prediction': list(prediction)})


@app.route('/upload', methods=['POST', 'GET'])
def upload_imagen():
    if request.method == 'POST' and 'imagen' in request.files:
        file = request.files['imagen']
        filename = photo.save(file)
        return filename
    return render_template('upload.html')


if __name__ == '__main__':
    # apenas inicie la app cargamos el modelo en memoria
    app.run()
