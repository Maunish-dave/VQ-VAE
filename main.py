from flask import Flask,request,render_template,send_file
from model.model import get_rec_image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

@app.route('/home',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        file  = request.files['file']
        filename = file.filename
        print(filename)
        if filename != '' and file:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'],filename)  
        get_rec_image(filename) 
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'],'image.jpg')  
        return render_template('home.html',file_path1=file_path,file_path2=file_path2)
    else:
        return render_template('home.html')

@app.route('/download',methods=['GET','POST'])
def download():
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpg')
    return send_file(path,as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

    