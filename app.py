from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return "No file"

    # Read the CSV file
    sleep_data = pd.read_csv(file)
    modified_sleep_data = sleep_data.drop(columns=['date'])
    X = modified_sleep_data.drop(columns=['Total Sleep Score'])
    y = modified_sleep_data['Total Sleep Score']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model and data for later use
    app.config['model'] = model
    app.config['X_columns'] = X.columns.tolist()
    app.config['sleep_data'] = sleep_data  # Save sleep_data in app.config

    # Render the table and input form
    table_html = sleep_data.to_html(classes='data').replace('\n', '')

    # Add hidden-rows class to rows beyond the first 10
    table_html = table_html.split('</tr>', 11)
    table_html[10] = table_html[10].replace('<tr', '<tr class="hidden-rows"')
    table_html = '</tr>'.join(table_html)

    return render_template('table.html', table=table_html, titles=sleep_data.columns.values)

@app.route('/predict', methods=['POST'])
def predict():
    model = app.config.get('model')
    X_columns = app.config.get('X_columns')
    sleep_data = app.config.get('sleep_data')

    if not model or not X_columns or sleep_data is None:
        return "Model not trained or columns not available"

    # Get user input data
    user_data = [float(request.form[col]) for col in X_columns]
    user_data = [user_data]  # Convert to 2D array

    # Make prediction
    prediction = model.predict(user_data)[0]

    # Generate the linear regression chart
    X = pd.DataFrame(user_data, columns=X_columns)
    y_pred = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(sleep_data[X_columns[2]], sleep_data['Total Sleep Score'], color='gray', label='Original Data')
    plt.scatter(X.iloc[:, 2], y_pred, color='blue', label='Predicted Data')
    plt.xlabel(X_columns[2])
    plt.ylabel('Total Sleep Score')
    plt.title('Linear Regression Prediction')
    plt.legend()
    chart_path = os.path.join('static', 'chart.png')
    plt.savefig(chart_path)
    plt.close()

    return render_template('result.html', prediction=prediction, chart_path=chart_path)

if __name__ == '__main__':
    app.run(debug=True)