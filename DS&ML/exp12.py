from sklearn.tree import DecisionTreeClassifier, plot_tree  # Import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt  # Import matplotlib for visualization

br = load_breast_cancer()
x = br.data
y = br.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
max_depth = 2

dc = DecisionTreeClassifier(max_depth=max_depth)
dc.fit(x_train, y_train)

p = dc.predict(x_test)
r = accuracy_score(y_test, p)
c = classification_report(y_test, p)

print("Accuracy: ", r)
print("\nClassification Report:\n", c)

plt.figure(figsize=(12, 8))
plot_tree(dc, filled=True, feature_names=br.feature_names, class_names=br.target_names, rounded=True)
plt.title("Decision Tree (Max Depth: {})".format(max_depth))
plt.show()
