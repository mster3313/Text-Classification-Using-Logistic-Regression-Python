from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def createBagOfWords(X_train, X_test):
    vectorizer = CountVectorizer(analyzer='word', input='content', stop_words='english', max_features=300)
    bag_of_words_train = vectorizer.fit_transform(X_train).toarray()
    bag_of_words_test = vectorizer.transform(X_test).toarray()
    return bag_of_words_train, bag_of_words_test


def createTFIDF(X_train, X_test):
    vectorizer = TfidfVectorizer(analyzer='word', input='content', stop_words='english', max_features=300)
    tfidf_train = vectorizer.fit_transform(X_train).toarray()
    tfidf_test = vectorizer.transform(X_test).toarray()
    return tfidf_train, tfidf_test