from flask import Flask, jsonify, render_template, send_file, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime, timedelta
import os

app = Flask(__name__)

df2 = pd.read_csv("data\Soho_earthquake_results_updated.csv")
threshold = 16.69
filtered_df2 = df2[df2['Np_max'] >= threshold]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_dates', methods=['GET'])
def get_dates():
    dates = filtered_df2[['Date', 'Np_max']].to_dict(orient='records')
    
    return jsonify(dates)


@app.route('/run_model', methods=['GET'])
def run_model():
    
    df = pd.read_csv('data/Soho_earthquake_results_updated_shifted.csv')

    X = df[['Np_max_normalized']]
    y = df['earthquakes_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    accuracy = accuracy_score(y_test, y_pred)

    return jsonify({
        'report': report,
        'accuracy': accuracy
    })

@app.route('/plot_graph', methods=['POST'])
def plot_graph():
    selected_date = request.json['selected_date']

    date_format = "%Y-%m-%d"
    selected_date_dt = datetime.strptime(selected_date.split(' - ')[0], date_format)
    date_range = [selected_date_dt - timedelta(days=4) + timedelta(days=i) for i in range(15)]
    date_range_str = [date.strftime(date_format) for date in date_range]
    day_data = df2[df2['Date'].isin(date_range_str)]

    if len(day_data) < 15:
        return jsonify({"error": "Not all days have data"}), 400

    Np_max_values = day_data['Np_max'].values
    earthquake_values = day_data['earthquakes'].values

    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    index = range(len(date_range_str))

    plt.bar(index, Np_max_values, color='blue', label='Np_max', width=bar_width, align='center')
    plt.bar([i + bar_width for i in index], earthquake_values, color='red', label='Earthquakes', width=bar_width, align='center')

    for i, v in enumerate(earthquake_values):
        plt.text(i + bar_width, v + 0.1, str(v), color='black', ha='center', va='bottom')

    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.title(f'Proton Density and Earthquakes from {date_range_str[0]} to {date_range_str[-1]}')
    plt.xticks([i + bar_width / 2 for i in index], date_range_str, rotation=45)
    plt.legend()
    plt.tight_layout()

    img_path = "static/graph.png"
    plt.savefig(img_path)
    plt.close()

    return jsonify({"img_path": img_path})

if __name__ == '__main__':
    app.run(debug=True)
