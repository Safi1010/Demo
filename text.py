import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ✅ Given dataset
data = np.array([
    [5, 3, 0], [4, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0],
    [10, 5, 1], [11, 3, 1], [12, 6, 1], [13, 7, 1], [14, 5, 1]
])

X_train = data[:, :2]  # Features (X, Y)
y_train = data[:, 2]   # Categories (0 or 1)

# ✅ Train KNN Classifier (K=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ✅ GUI Setup
root = tk.Tk()
root.title("KNN Classifier App")
root.geometry("600x600")
root.configure(bg="lightgray")

# ✅ User Input Fields
tk.Label(root, text="Enter X Coordinate:", font=("Arial", 12), bg="lightgray").pack(pady=5)
x_entry = tk.Entry(root, font=("Arial", 12))
x_entry.pack(pady=5)

tk.Label(root, text="Enter Y Coordinate:", font=("Arial", 12), bg="lightgray").pack(pady=5)
y_entry = tk.Entry(root, font=("Arial", 12))
y_entry.pack(pady=5)

# ✅ Plot Setup
fig, ax = plt.subplots(figsize=(5, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

def update_plot(new_points, new_labels):
    """ Updates the plot with new points and classification results """
    ax.clear()
    ax.set_title("2D Data Visualization with KNN Classification")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Plot category 0 (○ - Blue)
    ax.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
               marker='o', color='blue', label="Category 0 (○)")

    # Plot category 1 (× - Red)
    ax.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
               marker='x', color='red', label="Category 1 (×)")

    # Plot new points
    for (x, y), category in zip(new_points, new_labels):
        marker, color = ('*', 'green') if category == 0 else ('s', 'purple')
        ax.scatter(x, y, marker=marker, color=color, s=200, label=f"New Point ({x}, {y})")

    ax.legend()
    canvas.draw()

new_points = []
new_labels = []

def predict():
    """ Gets user input, classifies the new point, and updates the plot """
    try:
        X_new = float(x_entry.get())
        Y_new = float(y_entry.get())
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter numeric values!")
        return

    # Predict category using KNN
    new_point = np.array([[X_new, Y_new]])
    predicted_category = int(knn.predict(new_point)[0])

    # Store new point
    new_points.append((X_new, Y_new))
    new_labels.append(predicted_category)

    # Update plot
    update_plot(new_points, new_labels)
    messagebox.showinfo("Prediction", f"Predicted Category: {predicted_category}")

# ✅ Predict Button
tk.Button(root, text="Predict", font=("Arial", 12), bg="blue", fg="white", 
          command=predict).pack(pady=10)

# ✅ Start UI
update_plot(new_points, new_labels)  # Initialize with default dataset
root.mainloop()