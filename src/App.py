import json
from flask import Flask, render_template, request, send_from_directory
import os
from search_methods import *
app = Flask(__name__)


@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/result', methods=['POST','GET'])
def Result():
    if request.method == 'POST':
        if 'file' not in request.form:
            return render_template('index.html')
        
        Method = request.form['KNN']
        KValue = request.form['KValue']
        file = request.form['file']
        path = os.path.abspath('input/'+file)
    
        if Method == 'RTree':
            img = fr.load_image_file(path)
            query = fr.face_encodings(img)[0]
            data = KNN_rtree(int(KValue),query)
            return render_template('result1.html', result = data)
        if Method == 'Sequential':
            img = fr.load_image_file(path)
            query = fr.face_encodings(img)[0]
            data = KNN_Seq(int(KValue), query, 12800)
            return render_template('result2.html', result = data)

@app.route("/image/<directorio>/<filename>") 
def show_image(filename,directorio): 
    collection_path = '/mnt/d/UTEC/2021-2/BD2/Proyecto/Proyecto3_BD2_2021-2/DataProcessing/Collection/lfw/'
    directory = collection_path+directorio
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
