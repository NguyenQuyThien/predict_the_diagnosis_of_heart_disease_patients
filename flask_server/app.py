import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, jsonify, session
import pymysql
from app_config import app
import pickle
from db_config import mysql
from werkzeug import check_password_hash

import Normalization
import numpy as np

model = pickle.load(open('Model_Random_Forest.pkl', 'rb'))

# This section about admin
# Route for handling the login page logic
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        conn = None
        cursor = None
        username = request.form['username']
        password = request.form['password']
        try:
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("SELECT id, username, password FROM admin WHERE username=%s", username)
                row = cursor.fetchone()
                        
                if row:
                    if check_password_hash(row['password'], password):
                        # Create session data, we can access this data in other routes
                        session['loggedin'] = True
                        session['id'] = row['id']
                        session['username'] = row['username']
                        # Redirect to home page
                        return redirect(url_for('admin_home'))
                    else:
                        error = 'The password is incorrect. Please try again.'
                else:
                        error = 'Invalid Credentials. Please try again.'
        except Exception as e:
                        print(e)
        finally:
                cursor.close() 
                conn.close()
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))
'''
trang chu admin co chức năng thêm, xóa, logout
'''
@app.route('/admin-home', methods=['GET', 'POST'])
def admin_home():
    # Check if user is loggedin
    if 'loggedin' in session:
        if request.method == 'POST':
            if 'add_data' in request.form:
                return render_template('adminAddRecord.html', username=session['username'])
            elif 'delete_data' in request.form:
                return redirect(url_for('list_record'))
            elif 'logout' in request.form:
                return redirect(url_for('logout'))
        # User is loggedin show admin-home-page
        return render_template('adminHome.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
'''
chức năng thêm record
'''
@app.route('/add-record', methods=['GET', 'POST'])
def add_record():
    # Check if user is loggedin
    if 'loggedin' in session:
        if request.method == 'POST':
            if 'add-a-record' in request.form:
                float_features = [safe_cast(x, float) for x in request.form.values()]
                str_features = [safe_cast(x, str) for x in float_features]
                conn = None
                cursor = None
                try:
                        conn = mysql.connect()
                        cursor = conn.cursor(pymysql.cursors.DictCursor)
                        cursor.execute("INSERT INTO record (Age , Sex , Cp , Trestbps , Chol , Fbs , Restecg , Thalach , Exang , Oldpeak, Slope , Ca , Thal , Target , AdminID) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (str_features[0], str_features[1], str_features[2], str_features[3], str_features[4], str_features[5], str_features[6], str_features[7], str_features[8], str_features[9], str_features[10], str_features[11], str_features[12], str_features[13], session.get('id', 'didnt know adminID')))
                        conn.commit()
                        print('Just add a record has Admin.ID = {} and the features are:'.format(session.get('id', 'didnt know adminID')))
                        for x in range(14):
                            print(str_features[x])
                        return render_template('adminAddRecord.html', username=session['username'])
                except Exception as e:
                        print(e)
                finally:
                        cursor.close() 
                        conn.close()    
        # User is loggedin show admin-add-record
        return render_template('adminAddRecord.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

def safe_cast(val, to_type, default=np.NaN):
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default
'''
chức năng delete record >>> hien thi record list để chọn item record muốn xóa (/list-record & /delete-record)
'''
@app.route('/list-record', methods=['GET'])
def list_record():
    # Check if user is loggedin
    if 'loggedin' in session:
        conn = None
        cursor = None
        try:
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("SELECT * FROM record")
                records = cursor.fetchall()
                        
                if records:
                    return render_template('adminDeleteRecord.html', username=session['username'], records=records)
        except Exception as e:
                        print(e)
        finally:
                cursor.close() 
                conn.close()
        return render_template('adminDeleteRecord.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

@app.route('/delete-record', methods=['POST'])
def delete_record():
    # Check if user is loggedin
    if 'loggedin' in session:
        conn = None
        cursor = None
        id = request.form['record_ID_to_delete']
        try:
                conn = mysql.connect()
                cursor = conn.cursor(pymysql.cursors.DictCursor)
                cursor.execute("DELETE FROM record WHERE id=%s", id)
                conn.commit()
                print('Just delete record has ID = ', id)
        except Exception as e:
                print(e)
        finally:
                cursor.close() 
                conn.close()
        return redirect(url_for('list_record'))
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
    
# the rest (to end this file) is about 'user' route
'''
trang chu user chức năng xem chuẩn đoán
'''
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]

    max_value = [float(77), float(1), float(4), float(200), float(603), float(1), float(2), float(202), float(1), float(6.2), float(3), float(9), float(7)]
    min_value = [float(28), float(0), float(1), float(0), float(0), float(0), float(0), float(60), float(0), float(-2.6), float(0), float(0), float(1)]
    final_features.append(max_value)
    final_features.append(min_value)
    
    temp_scales = Normalization.rescaling_with_MinMaxScaler_sklearn(final_features)
    
    prediction = model.predict([np.array(temp_scales[0])])

    output = prediction[0]
    if (output == 1):
        ketluan = "có"
    else: ketluan = "không có"
    return render_template('index.html', prediction_text='bệnh nhân {} nguy cơ mắc bệnh lý tim mạch...'.format(ketluan))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
