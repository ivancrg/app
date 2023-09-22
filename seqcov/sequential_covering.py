import pandas as pd
from sklearn import tree
from learn_one_rule import LearnOneRule
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SequentialCovering(BaseEstimator, ClassifierMixin):

    def __init__(
            self,
            data,
            multiclass=False,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=None,
            max_leaf_nodes=None,
            output_name='Postoperative diagnosis'):
        self.data_orig = data
        self.data_orig.reset_index(drop=True, inplace=True)
        self.multiclass = multiclass
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.output_name = output_name
        self.result = None

    def remove_covered_instances(self, data, rule):
        data.reset_index(drop=True, inplace=True)

        # print(f'Data length: {len(data)}')

        # Creating condition for removing instances
        # (all instances under remove condition should be removed)
        remove_condition = pd.Series([True] * len(data))
        for condition in rule:
            feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

            if op == '<=':
                remove_condition &= (data[feat] <= thr)
            elif op == '>':
                remove_condition &= (data[feat] > thr)
            else:
                print("main.py::remove_covered_instances WARNING: Unknown operator!")

        # Filtering instances which should be removed (because they're covered by the new_rule)
        new_data = data[~remove_condition]

        return new_data

    def fit(self):
        if self.multiclass:
            self.result = self.sc_multiclass(self.data_orig.copy())
        else:
            self.result = self.sc(self.data_orig.copy())

    def sc(self, data, class_names=['0', '1']):
        rules_preds = []

        while len(data) > 0:
            data.reset_index(drop=True, inplace=True)

            lor = LearnOneRule(
                data.copy(),
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                output_name=self.output_name,
                class_names=class_names)
            _, new_rule, pred, n_covered = lor.learn_one_rule(
                0, None, None, None, [])
            # lor.plot_classifier()
            # plt.show()

            # Append a rule as a tuple of an array of (feature, value) conditions and outcome
            rules_preds.append((new_rule, pred))

            # Removing covered instances
            data = self.remove_covered_instances(data, new_rule)

        return rules_preds

    def sc_multiclass(self, data):
        # Array of tuples - (class, number of instances of that class)
        y = pd.DataFrame(data.iloc[:, -1])
        classes_counts = [(c, sum(y.iloc[:, 0] == c))
                          for c in y.iloc[:, 0].unique()]
        classes_counts.sort(key=lambda x: x[1])

        rules = []

        # print(classes_counts, y.value_counts(ascending=True))

        # While there are classes for which the rule hasn't been set yet
        while len(classes_counts) > 1:
            data.reset_index(drop=True, inplace=True)
            # print(data[self.output_name].value_counts())
            # Current class for which a rule should be set
            current_class = classes_counts[0][0]

            # print(f'Current class: {current_class}')

            # Copy of the dataset to be modified to binary classification
            data_current = data.copy()
            data_current.loc[data.iloc[:, -1] ==
                             current_class, self.output_name] = 1
            data_current.loc[data.iloc[:, -1] !=
                             current_class, self.output_name] = 0

            # Calculating the rule for the current class
            rules_preds_bin = self.sc(data_current, class_names=['other', current_class.astype(str)])

            # Remove all covered (positive) instances using rules_binary
            data = self.reconfigure_data_mc(data, rules_preds_bin)
            # for r, pred in rules_preds_bin:
            #     # We remove only positive data instances
            #     if pred == 1:
            #         print(f'Removing instances for class {current_class} using rule {r}')
            #         data = self.remove_covered_instances(data, r)

            # Add the new rules to result
            rules.append(
                {'class': current_class, 'rules_preds': rules_preds_bin})

            # Removing the class for which the rule has been calculated
            classes_counts.remove(classes_counts[0])

        # Adding a default rule
        rules.append({
            'class': classes_counts[0][0],
            'rules_preds': [([{
                'feature': 'default',
                'operator': '',
                'threshold': ''
            }], 1)]})

        return rules

    def predict_tmp(self, input):
        if self.multiclass:
            return self.predict_mc(self.result, input)
        else:
            return self.predict_binary(self.result, input)
        
    def predict(self, input):
        if self.multiclass:
            return np.array(self.predict_mc(self.result, input)['Prediction'].to_list())
        else:
            return np.array(self.predict_binary(self.result, input)['Prediction'].to_list())

    def predict_binary(self, result, input):
        input.loc[:, 'Prediction'] = "N/A"

        for rule_pred in result:
            rule, pred = rule_pred

            # Keeps the record of all instances valid for
            # assigning a prediction
            pred_condition = (input['Prediction'] == "N/A")

            for condition in rule:
                feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

                if op == '<=':
                    pred_condition &= (input[feat] <= thr)
                elif op == '>':
                    pred_condition &= (input[feat] > thr)
                else:
                    print(
                        f"main.py::predict WARNING: Unknown operator! {feat} {op} {thr} {len(rule)}")

            input.loc[pred_condition, 'Prediction'] = pred

        return input
    
    def reconfigure_data_mc(self, input, rules_preds):
        new_data = input.copy()
        input.loc[:, 'Remove'] = "N/A"

        for rule, pred in rules_preds:
            if len(rule) == 0 or (len(rule) == 1 and rule[0]['feature'] == 'default'):
                if pred == 1:
                    valid = (input['Remove'] == "N/A")
                    input.loc[valid, 'Remove'] = 1
                continue

            pred_condition = (input['Remove'] == "N/A")

            for condition in rule:
                feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

                if op == '<=':
                    pred_condition &= (input[feat] <= thr)
                elif op == '>':
                    pred_condition &= (input[feat] > thr)
                else:
                    print(
                        f"main.py::reconfigure_data_mc WARNING: Unknown operator! {feat} {op} {thr} {len(rule)}")

            input.loc[pred_condition, 'Remove'] = pred

        return new_data[input['Remove'] != 1]

    def predict_mc(self, clf, input):
        input.loc[:, 'Prediction'] = "N/A"

        for c in clf:
            current_class = c['class']

            # Keeps the record of all instances valid for
            # assigning a prediction of current class
            # (e.g. if a prediction is 0, the valid values of
            #  appropriate instances will be set to False)
            valid = (input['Prediction'] == "N/A")

            for rule, pred in c['rules_preds']:
                if len(rule) == 0 or (len(rule) == 1 and rule[0]['feature'] == 'default'):
                    if pred == 1:
                        input.loc[valid, 'Prediction'] = current_class
                    continue

                pred_condition = (input['Prediction'] == "N/A")

                for condition in rule:
                    feat, op, thr = condition['feature'], condition['operator'], condition['threshold']

                    if op == '<=':
                        pred_condition &= (input[feat] <= thr)
                    elif op == '>':
                        pred_condition &= (input[feat] > thr)
                    else:
                        print(
                            f"main.py::predict WARNING: Unknown operator! {current_class} {feat} {op} {thr} {len(rule)}")

                if pred == 0:
                    # If the prediction is 0, that instances are considered
                    # not members of the current_class therefore we do not
                    # consider them during evaluation of subsequent rules
                    valid &= (~pred_condition)
                else:
                    input.loc[(valid & pred_condition),
                              'Prediction'] = current_class

        return input

    def print_rp_binary(self, rules_preds):
        for rule, pred in rules_preds:
            text = ' AND '.join(
                [f"{condition['feature']} {condition['operator']} {condition['threshold']}" for condition in rule])
            print(f"For rule {text} --> Predicted class: {pred}")

    def print_rules_preds(self):
        if self.multiclass:
            for res in self.result:
                print(f"Rules for class {res['class']}:")
                self.print_rp_binary(res['rules_preds'])
                print()
        else:
            self.print_rp_binary(self.result)
