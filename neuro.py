import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.DataFrame({
    "Экранные часы": [
        8,6,10,3,2,7,9,4,5,6,
        12,11,9,2,1,13,7,6,4,5,
        3,8,10,9,12,2,5,6,11,7
    ],
    "Часы сна": [
        5,6,4,9,8,6,5,8,7,7,
        4,5,5,9,10,3,6,7,8,7,
        9,5,4,6,4,10,8,7,5,6
    ],
    "Какой в итоге пронулся": [
        "уставший","отдохнувший","уставший","отдохнувший","отдохнувший","уставший","уставший","отдохнувший","отдохнувший","отдохнувший",
        "уставший","уставший","уставший","отдохнувший","отдохнувший","уставший","уставший","отдохнувший","отдохнувший","отдохнувший",
        "отдохнувший","уставший","уставший","уставший","уставший","отдохнувший","отдохнувший","отдохнувший","уставший","уставший"
    ]
})

X = data[["Экранные часы","Часы сна"]].values
y = data["Какой в итоге пронулся"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=15)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test,y_pred):.2f}")

def main():
    screen = int(input("Введите кол-во часов у экрана: "))
    sleep = int(input("Введите количество часов сна: "))
    sample = np.array([[screen,sleep]])
    result = clf.predict(sample)
    print("Ты будешь",result[0])

if __name__ == "__main__":
    main()
