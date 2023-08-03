import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

data= pd.read_csv("weather.csv")
columns = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RISK_MM'] 

x = data[columns]       
x = x.fillna(0)
y = data.RainTomorrow 


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

classifier = LogisticRegression(solver='liblinear')
classifier.fit(xtrain, ytrain)
y_pred = classifier.predict(xtest)

cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)

print ("Accuracy : ", accuracy_score(ytest, y_pred))

class_names = [0,1]
fig, ax = plt.subplots() 

tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names) 

plt.yticks(tick_marks, class_names) 

sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top") 
plt.tight_layout() 

plt.title('Confusion matrix', y=1.1) 

plt.ylabel('Actual label ') 

plt.xlabel('Predicted label')
plt.show()