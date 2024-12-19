import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

data = pd.read_csv("pushup_data.csv")

data['Elbow_Asymmetry'] = abs(data['Elbow_Left'] - data['Elbow_Right'])
data['Wrist_Asymmetry'] = abs(data['Wrist_Left'] - data['Wrist_Right'])

data['Average_Asymmetry'] = (data['Elbow_Asymmetry'] + data['Wrist_Asymmetry']) / 2

plt.figure(figsize=(12, 6))
plt.plot(data['Rep'], data['Elbow_Asymmetry'], label='Elbow Asymmetry', marker='o')
plt.plot(data['Rep'], data['Wrist_Asymmetry'], label='Wrist Asymmetry', marker='x')
plt.plot(data['Rep'], data['Average_Asymmetry'], label='Average Asymmetry', linestyle='--')
plt.xlabel("Repetition")
plt.ylabel("Asymmetry Score")
plt.title("Kinematic Asymmetry Over Repetitions")
plt.legend()
plt.grid()
plt.show()

fatigue_threshold_rep = data['Rep'].iloc[-1]  
correlation_elbow = linregress(data['Rep'], data['Elbow_Asymmetry'])
correlation_wrist = linregress(data['Rep'], data['Wrist_Asymmetry'])

print("Elbow Asymmetry Correlation:")
print(f"Slope: {correlation_elbow.slope}, R-squared: {correlation_elbow.rvalue ** 2}")
print("\nWrist Asymmetry Correlation:")
print(f"Slope: {correlation_wrist.slope}, R-squared: {correlation_wrist.rvalue ** 2}")

data.to_csv("processed_pushup_data.csv", index=False)
print("Processed data saved to 'processed_pushup_data.csv'")
