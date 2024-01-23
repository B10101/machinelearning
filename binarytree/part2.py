import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import graphviz

# Task 1: Restructure the dataset
dataset = pd.read_csv('lab01_dataset_2.csv')
X = dataset.drop('Output', axis=1)  # Replace 'target_class' with the actual target column name
y = dataset['Output']
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X.select_dtypes(include=['object'])).toarray()
X_encoded = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns))
X_final = pd.concat([X.select_dtypes(exclude=['object']), X_encoded], axis=1)

# Task 2: Perform supervised learning using decision tree classifiers
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Task 3: Display the results by predicting the class of the test set
y_pred = clf.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Results:\n", results)

# Task 4: Output the decision tree
text_representation = export_text(clf)
print("Decision Tree:\n", text_representation)

dot_data = export_graphviz(clf, out_file=None, 
                           feature_names=X_final.columns,
                           class_names=clf.classes_,
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_graphical")
graph.view("decision_tree_graphical")
