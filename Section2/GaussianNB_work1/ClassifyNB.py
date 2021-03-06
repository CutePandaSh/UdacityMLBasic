def classify(features_train, labels_train):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier


    ### your code goes here!

    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', gamma=5.0, C= 5.0)
    clf.fit(features_train, labels_train)
    return clf