from __future__ import print_function

import numpy as np
import os
from flask import *
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'C:/Users/Aditya Kyatham/AppData/Local/Programs/Python/Python36/FINAL_BE_PROJ/NSL-KDD/uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
from sklearn.externals import joblib

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras import backend as K
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.models import model_from_json

json_file = open('C:/Users/Aditya Kyatham/AppData/Local/Programs/Python/Python36/FINAL_BE_PROJ/NSL-KDD/cnn_model.json', 'r')
cnn_model_json = json_file.read()
json_file.close()
#cnn_model = model_from_json(cnn_model_json)
# load weights into new model


# evaluate loaded model on test data
#score = loaded_model.evaluate(X, Y, verbose=0)

knn_model = joblib.load('C:/Users/Aditya Kyatham/AppData/Local/Programs/Python/Python36/FINAL_BE_PROJ/NSL-KDD/knn_model_anomaly.sav')
@app.route('/')
def home():
	return render_template('index.html')


def allowed_file(filename):
	return '.' in filename and \
    		filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict',methods=['POST'])

def predict():
	'''
	For rendering results on HTML GUI
	'''
	if 'test_data' not in request.files:
		flash('No Data found')
		return redirect(request.url)
	file = request.files['test_data']
        # if user does not select file, browser also
        # submit an empty part without filename
	if file.filename == '':
		flash('No selected file')
		return redirect(request.url)	
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
       # return redirect(url_for('uploaded_file',filename=filename))
    
	test_data= pd.read_csv('C:/Users/Aditya Kyatham/AppData/Local/Programs/Python/Python36/FINAL_BE_PROJ/NSL-KDD/uploads/'+filename, index_col=None)
	from sklearn.preprocessing import LabelEncoder
	encodings = dict()
	for c in test_data.columns:
    #print df[c].dtype
		if test_data[c].dtype == "object":

			encodings[c] = LabelEncoder() #to give numerical label to char type labels.
			encodings[c]
			test_data[c] = encodings[c].fit_transform(test_data[c])
    
	C = test_data.iloc[:,41]
	T = test_data.iloc[:,0:41]
	T=T.drop(['land','su_attempted','num_outbound_cmds','is_host_login','urgent','num_failed_logins','su_attempted','num_file_creations','num_shells','srv_diff_host_rate'], axis = 1) 
	from sklearn.preprocessing import StandardScaler #normalization
	testT= StandardScaler().fit_transform(T)
	testT_nn = np.array(T)
	testT_nn.astype(float)

	scaler = Normalizer().fit(testT_nn)
	testT_nn = scaler.transform(testT_nn)
	y_test = np.array(C)
	y_test1= to_categorical(y_test)
	X_test = np.array(testT)##
	X_test_nn = np.array(testT_nn)##
	batch_size = 64
	X_test1_nn = np.reshape(X_test_nn, (X_test_nn.shape[0],X_test_nn.shape[1],1))
	cnn_model = model_from_json(cnn_model_json)
	cnn_model.load_weights("C:/Users/Aditya Kyatham/AppData/Local/Programs/Python/Python36/FINAL_BE_PROJ/NSL-KDD/cnn_model.h5")
	cnn_model._make_predict_function()
	#cnn_model._make_predict_function()
	from sklearn.metrics import accuracy_score 

	pred_cnn = cnn_model.predict_classes(X_test1_nn)
	pred_cnn1 = pred_cnn
	pred_cnn_prob = cnn_model.predict_proba(X_test1_nn, batch_size = 64)
	#cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	#loss, accuracy_cnn = cnn_model.evaluate(X_test1_nn, y_test1)
	accuracy_cnn = round(accuracy_score(y_test, pred_cnn)*100,2)

	error_index_nn=[]
	error_index_nn_X=[]
	error_index_nn_Y=[]
	for i in range(pred_cnn_prob.shape[0]):
		pred_cnn_prob[i].sort()
		if abs(pred_cnn_prob[i][4]-pred_cnn_prob[i][3])<0.96:
			error_index_nn.append(i)
			error_index_nn_X.append(X_test[i])
			error_index_nn_Y.append(y_test[i])
	error_index_nn_X1 = np.array(error_index_nn_X)
	from collections import Counter
	pred_knn_whole = knn_model.predict(X_test)

	accuracy_knn = round(accuracy_score(y_test, pred_knn_whole)*100,2)
	if error_index_nn_X1!= []:
		pred_knn_prob = knn_model.predict_proba(error_index_nn_X1)
		pred_knn = knn_model.predict(error_index_nn_X1)
		i=0
		for data in error_index_nn:

			pred_knn_prob[i].sort()
			if abs(pred_knn_prob[i][4]-pred_knn_prob[i][3])>0.00:
				pred_cnn1[data]=pred_knn[i]
  			
			i+=1
		attacks = pred_cnn1.shape[0]- Counter(pred_cnn1).get(1)
		accuracy = round(accuracy_score(y_test, pred_cnn1)*100,2)
	else:
		
		accuracy=accuracy_cnn
		attacks = pred_cnn1.shape[0]- Counter(pred_cnn1).get(1)
	
	
	pred_cnn1 = encodings[c].inverse_transform(pred_cnn1)
	np.savetxt("C:/Users/Aditya Kyatham/AppData/Local/Programs/Python/Python36/FINAL_BE_PROJ/NSL-KDD/Predictions/Predictions.csv",pred_cnn1, delimiter=",",fmt='%s')
	







	

	
	

	

	

	return render_template('index.html', prediction_text='{} attacks predicted by our model & Accuracy is:-  Our Model: {},CNN: {},KNN: {}'.format(attacks,accuracy,accuracy_cnn,accuracy_knn))

if __name__ == "__main__":
	app.run(debug=True)