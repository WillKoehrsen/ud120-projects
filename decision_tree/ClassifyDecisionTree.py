def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    from sklearn import tree

    clf = tree.DecisionTreeClassifier # create the classifier
    clf = clf.fit(features_train, labels_train) # train the classifier

    return clf