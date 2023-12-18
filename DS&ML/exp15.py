from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score

category = ['alt.atheism','soc.religion.christian','comp.graphics','sci.med']
t = fetch_20newsgroups(subset='train',categories=category,shuffle=True,random_state=42)

func=TfidfVectorizer()
an = func.fit_transform(t.data)    #Tfid of the feature
print(an)

y = t.target
x_train,x_test,y_train,y_test=train_test_split(an,y,test_size=0.2,random_state=42)
svm_classifier=SVC(kernel='linear',random_state=42)
svm_classifier.fit(x_train,y_train)
pred=svm_classifier.predict(x_test)
acc_score=accuracy_score(y_test,pred)

class_repo=classification_report(y_test,pred,target_names=t.target_names)
print("accuracy_score",acc_score)
print("classification_report\n",class_repo)

new_data = ["computer graphics"]

x_new_tfid=func.transform(new_data)
new_pred =svm_classifier.predict(x_new_tfid)

for i,text in enumerate(new_data):
    predicted_category = t.target_names[new_pred[i]]
    print("predicted category ",predicted_category)