import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Đọc dữ liệu ---
df = pd.read_csv("university_admission_single_input.csv")
X_raw = df[['hours_studied']].values
y = df['admitted'].values

# --- 2. Chuẩn hóa dữ liệu ---
mu = np.mean(X_raw)
sigma = np.std(X_raw)
X_std = (X_raw - mu) / sigma

# Thêm bias
X = np.hstack([np.ones((X_std.shape[0], 1)), X_std])  # shape (200, 2)

# --- 3. Định nghĩa các hàm ---
def sigmoid(z): # Hàm sigmoid
    return 1 / (1 + np.exp(-z))

def compute_loss(y, y_pred): # Hàm mất mát 
    eps = 1e-9
    return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

def train_logistic(X, y, lr=0.1, epochs=1000): # Hàm huấn luyện
    m, n = X.shape
    weights = np.zeros(n)
    for epoch in range(epochs):
        z = X @ weights
        y_pred = sigmoid(z)
        gradient = X.T @ (y_pred - y) / m # Gradient Descent
        weights -= lr * gradient          # Cập nhật trong số (Gradient Descent)
    return weights

def predict(x_input, weights):
    return sigmoid(x_input @ weights)

# --- 4. Huấn luyện mô hình ---
weights = train_logistic(X, y)
y_pred_train = (sigmoid(X @ weights) >= 0.5).astype(int)
accuracy = np.mean(y_pred_train == y)

# --- 5. Dự đoán điểm mới ---
new_value = 2  #  Giá trị cần dự đoán
new_std = (new_value - mu) / sigma
new_point = np.array([1, new_std])  # [bias, x]
new_pred = predict(new_point, weights)

# --- 6. Vẽ đồ thị ---
x_vals = np.linspace(0, 12, 300)
x_vals_std = (x_vals - mu) / sigma
X_plot = np.c_[np.ones_like(x_vals_std), x_vals_std]
y_probs = predict(X_plot, weights)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_probs, label='Đồ thị dự đoán', color='blue')

# Dữ liệu thực tế
plt.scatter(X_raw[y == 0], y[y == 0], color='red', label='Trượt')
plt.scatter(X_raw[y == 1], y[y == 1], color='green', label='Đỗ')

# Điểm mới (vàng)
plt.scatter([new_value], [new_pred], color='yellow', edgecolor='black', s=100, label='Điểm cần dự đoán')
plt.text(new_value + 0.2, new_pred, f"({new_value:.1f}, {new_pred:.2f})", color='black')

# Hiện độ chính xác
plt.text(0, -0.1, f"Độ chính xác: {accuracy:.2%}", fontsize=12, ha='left', color='black')

# Trang trí
plt.xlabel("Số giờ học")
plt.ylabel("Khả năng trúng tuyển")
plt.ylim(-0.2, 1.1)
plt.title("Mô hình dự đoán trúng tuyển Đại học")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
 # type: ignore