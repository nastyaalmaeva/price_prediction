import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

os.makedirs('../plots', exist_ok=True)

X_test = np.load('../data/X_test.npy')
y_test = np.load('../data/y_test.npy')
scaler = joblib.load('../models/scaler.pkl')
model = load_model('../models/lstm_model.keras')

predictions_scaled = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

mae = mean_absolute_error(y_test_actual, predictions)
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

print(f'mae: {mae:.2f}, rmse: {rmse:.2f}')

plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='actual', color='blue')
plt.plot(predictions, label='predicted', color='red', alpha=0.7)
plt.xlabel('days')
plt.ylabel('price ($)')
plt.legend()
plt.grid(True)
plt.savefig('../plots/predictions_vs_actual.png', dpi=300)
plt.close()

print('plot saved to plots/predictions_vs_actual.png')