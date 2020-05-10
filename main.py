from flask import Flask, url_for, request, render_template, jsonify
import sklearn
import joblib
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/linearRegression', methods=['POST'])
def get_samples():
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 0
    f5 = 0
    action = request.form['action']
    adventure = request.form['adventure']
    puzzle = request.form['puzzle']
    rpg = request.form['rpg']
    strategy = request.form['strategy']
    if(action == 'true'): f1 = 1
    if(adventure == 'true'): f2 = 1
    if(puzzle == 'true'): f3 = 1
    if(rpg == 'true'): f4 = 1
    if(strategy == 'true'): f5 = 1
    print(f1, f2, f3, f4, f5)
    model = joblib.load('./finalized_model.pkl')
    answer = model.predict(pd.DataFrame({'Action':[f1], 'Adventure':[f2], 'Puzzle':[f3], 'RPG':[f4],'Strategy':[f5]}))
    return jsonify(str(answer[0]))

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host="127.0.0.1", port=8080, debug=True)
