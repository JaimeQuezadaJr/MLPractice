from flask import Flask, request, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    # Make predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return f"Mean Squared Error: {mse}"

if __name__ == '__main__':
    app.run(debug=True)