
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# 创建scaler对象，用户可选择缩放的区间，一般选择[0,1]

scaler = MinMaxScaler(feature_range=(0, 1))
X_train = np.random.randn(100, 2)

# 拟合训练集

scaler.fit(X_train)

# 缩放训练集

X_train_scaled = scaler.transform(X_train)

# 缩放检验集

X_test = np.array([

[185, 0.25],

[150, 0.55]

])

X_test_scaled = scaler.transform(X_test)

# 查看结果

print(X_train)

print(X_train_scaled)

print(X_test_scaled)