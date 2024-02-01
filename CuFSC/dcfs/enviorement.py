import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.utils import to_categorical

class Enviorement:

    def __init__(self, dataset, learner_model):
        self.data = dataset
        self.action_space = np.zeros(self.data.shape[1] - 1)
        self.data_size = len(self.data)
        self.learner_model = learner_model
        self.number_of_classes = self.data.iloc[:, -1].unique().shape[0]

    # Function that create the episode data - sample randomaly (need to add boosting)
    def get_data(self, episode_size, for_episode, mode):
        global dataset
        if mode == 'train':
            if for_episode == 1:
#                 dataset = self.data.sample(n=episode_size, replace=False)
                dataset = self.data.sample(n=episode_size, replace=False)#, random_state=1)

#                 print(f"episode size {episode_size}")
#                 X_train, X_test, y_train, y_test = train_test_split(self.data.iloc[:, 0:-1], self.data.iloc[:, -1],
#                                                                     stratify=self.data.iloc[:, -1],
#                                                                     test_size=max(0, self.data.shape[0]-episode_size))
#                 dataset = pd.concat([X_train, y_train], axis=1)
            else:
                dataset = self.data
        else:
            dataset = self.test_data
        return dataset

    def data_separate(self, dataset):
        global X
        global y
        X = dataset.iloc[:, 0:dataset.shape[1] - 1]  # all rows, all the features and no labels
        y = dataset.iloc[:, -1]  # all rows, label only
        return X, y

    # Function that split the episode data into train and test
    def data_split(self, X, y):
        from sklearn.model_selection import train_test_split
#         X_train_main, X_test_main, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=4)
        X_train_main, X_test_main, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        return X_train_main, X_test_main, y_train, y_test

    def s2(self, s, a, selected_actions):
        s2 = to_categorical(a, len(self.action_space))
        sel_actions = selected_actions.copy()
        sel_actions[a] = 1
        return s2, sel_actions

    def accuracy(self, s2, X_train_main, X_test_main, y_train, y_test):
        columns = np.where(s2 == 1)[0]
        X_train = X_train_main.iloc[:, columns]
        X_test = X_test_main.iloc[:, columns]
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        accuracy = self.leraner(X_train, X_test, y_train, y_test)
        return accuracy

    def leraner(self, X_train, X_test, y_train, y_test):
        if self.learner_model == 'DT':
            learner = tree.DecisionTreeClassifier()
            learner = learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
        elif self.learner_model == 'NB':
            learner = MultinomialNB()
            learner = learner.fit(X_train, y_train)
            y_pred = learner.predict(X_test)
        accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
#         accuracy = metrics.accuracy_score(y_test, y_pred)

        return accuracy