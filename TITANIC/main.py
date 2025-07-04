'''
Задача классификации набор данных титаник 
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

#print(data_train.head())

classes = ['Pclass', 'Sex', 'SibSp', 'Parch']


#input data
y = data_train['Survived']

#output data
x = pd.get_dummies(data_train[classes])
x_test = pd.get_dummies(data_test[classes])

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=1,)
model.fit(x, y)


prediction = model.predict(x_test)

output = pd.DataFrame({'PassengerId': data_test['PassengerId'], 'Survived': prediction})
output.to_csv('submission.csv', index=False)

