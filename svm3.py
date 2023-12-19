import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("./csv/student_data_.csv")

# Split data into features (X) and target variable (y)
X = data[["Study Hours", "Sleep Hours", "Play Game Hours"]]
y = data["Pass Exam"]

# Split data into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict on the Test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Input data for 20 students at once through a CSV file
input_file_path = "./csv/input_data.csv"
input_data = pd.read_csv(input_file_path)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Make predictions for each student and print the results
for i in range(len(input_data)):
    study_hours = input_data.loc[i, "Study Hours"]
    sleep_hours = input_data.loc[i, "Sleep Hours"]
    gaming_hours = input_data.loc[i, "Play Game Hours"]
    student_name = input_data.loc[i, "Name"]

    # Create DataFrame from input data
    input_data_df = pd.DataFrame(
        {
            "Study Hours": [study_hours],
            "Sleep Hours": [sleep_hours],
            "Play Game Hours": [gaming_hours],
        }
    )

    # Predict the exam result
    prediction = model.predict(input_data_df)

    # Print the results
    result = "Pass" if prediction[0] == 1 else "Fail"
    print(f"name: {student_name}, exam: {result}")

    # Plot the results on a 3D scatter plot
    color = "green" if prediction[0] == 1 else "red"
    ax.scatter(study_hours, sleep_hours, gaming_hours, c=color)

# Display the hyperplane with a lower alpha value for a thinner appearance
xx, yy = np.meshgrid(X.iloc[:, 0], X.iloc[:, 1])
zz = (
    -model.intercept_[0] - model.coef_[0, 0] * xx - model.coef_[0, 1] * yy
) / model.coef_[0, 2]
ax.plot_surface(xx, yy, zz, alpha=0.1, color="blue", label="Hyperplane")

# Set axis labels
ax.set_xlabel("Study Hours")
ax.set_ylabel("Sleep Hours")
ax.set_zlabel("Play Game Hours")
ax.set_title("SVM classifier")

# Display the 3D plot
plt.show()
