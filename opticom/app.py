from flask import Flask, request, jsonify
import gazer  # Import your gaze-tracking logic
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "OptiCom Backend is Running"

@app.route('/base')
def show_base():
    return render_template('base.html')

@app.route('/userprofile')
def show_userprofile():
    return render_template('userprofile.html')

@app.route('/history')
def show_history():
    return render_template('history.html')

@app.route('/logout')
def logout():
    return "Logged out"  # Replace with your logout logic


@app.route('/calibrate')
def show_calibrate():
    return render_template('calibrate.html')

@app.route('/train')
def show_train():
    return render_template('train.html')

@app.route('/predict')
def show_predict():
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)


@app.route('/calibrate', methods=['POST'])
def calibrate():
    data = request.json
    result = gazer.calibrate(data['iris_data'], data['screen_data'])
    return jsonify({"status": "success", "message": "Calibration complete", "result": result})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    iris_position = data['iris_position']
    predicted_screen = gazer.predict(iris_position)
    return jsonify({"predicted_screen": predicted_screen})

@app.route('/')
def home():
    return "OptiCom Backend is Running"

if __name__ == '__main__':
    app.run(debug=True)
