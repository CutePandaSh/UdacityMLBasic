#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

print 'get data'
### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=20, p=2)
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier()
# from sklearn.svm import SVC
# clf = SVC(kernel= 'linear')
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# from sklearn.ensemble import RandomForestClassifier as RFC
# clf = RFC(n_estimators=10, min_samples_split= 10)
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier()
print "Starting trainning:"
t0 = time()
clf.fit(features_train, labels_train)
print "Finish training spend: ", time() - t0,"s"

print 'Starting predict:'
t0 = time()
pred = clf.predict(features_test)
print "Finish predict spend: ", time() - t0, "s"

print 'Accuracy is : ', accuracy_score(pred, labels_test)





# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass
