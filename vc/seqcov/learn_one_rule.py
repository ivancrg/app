import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np


class LearnOneRule:
    def __init__(
            self,
            data,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_leaf_nodes=None,
            output_name='Postoperative diagnosis',
            class_names=['0', '1']):
        self.data = data
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.output_name = output_name

        X, y = self.data.iloc[:, :-1], pd.DataFrame(self.data.iloc[:, -1])

        self.feature_names = X.columns
        self.class_names = class_names

        self.clf = tree.DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=42)
        self.clf = self.clf.fit(X.to_numpy(), y.to_numpy())

        # print(self.data[self.output_name].value_counts())
        # print(self.class_names)
        # print(self.clf.classes_)

    def plot_classifier(self):
        plt.figure(figsize=(19, 11))
        tree.plot_tree(self.clf, filled=True,
                       feature_names=self.feature_names, class_names=self.class_names)

    # Finds the decision tree's node with the minimum gini value
    # Returns the node's gini value, the node's decision path (list),
    # the predicted class and the number of covered samples
    def learn_one_rule(self, id, cond_feature, cond_operator, cond_threshold, rules):
        id_left, id_right = self.clf.tree_.children_left[id], self.clf.tree_.children_right[id]
        current_feature = self.clf.tree_.feature[id]
        current_threshold = self.clf.tree_.threshold[id]

        if cond_feature is not None:
            rules.append({
                'feature': cond_feature,
                'operator': cond_operator,
                'threshold': cond_threshold
            })

        # Leaf node
        if id_left == -1 and id_right == -1:
            # print(id, self.clf.tree_.impurity[id], rules, self.clf.tree_.value[id], np.argmax(
            #     self.clf.tree_.value[id]), self.clf.tree_.n_node_samples[id])
            return [self.clf.tree_.impurity[id], rules, np.argmax(self.clf.tree_.value[id]), self.clf.tree_.n_node_samples[id]]

        # Propagate to left and right nodes
        mgn_left, rules_left, class_left, covered_left = self.learn_one_rule(
            self.clf.tree_.children_left[id], self.feature_names[current_feature], '<=', current_threshold, rules.copy())
        mgn_right, rules_right, class_right, covered_right = self.learn_one_rule(
            self.clf.tree_.children_right[id], self.feature_names[current_feature], '>', current_threshold, rules.copy())

        if mgn_left < mgn_right or (mgn_left == mgn_right and covered_left >= covered_right):
            return [mgn_left, rules_left, class_left, covered_left]
        else:
            return [mgn_right, rules_right, class_right, covered_right]
