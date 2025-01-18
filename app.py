from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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

    # Render the table and input form
    table_html = sleep_data.to_html(classes='data').replace('\n', '')

    return render_template('table.html', tables=table_html, titles=sleep_data.columns.values)

@app.route('/predict', methods=['POST'])
def predict():
    model = app.config.get('model')
    X_columns = app.config.get('X_columns')

    if not model or not X_columns:
        return "Model not trained or columns not available"

    # Get user input data
    user_data = [float(request.form[col]) for col in X_columns]
    user_data = [user_data]  # Convert to 2D array

    # Make prediction
    prediction = model.predict(user_data)

    return f"Predicted Sleep Score: {prediction[0]}"

if __name__ == '__main__':
    app.run(debug=True)