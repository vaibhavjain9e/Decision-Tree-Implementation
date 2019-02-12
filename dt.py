import numpy as np
import utils as Util
from collections import Counter
from collections import deque


class DecisionTree:
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
        return y_pred

    def add_real_predicted_label_property(self, X_test, y_test):
        self.root_node.optimize_tree(X_test, y_test)


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        self.parent = None
        self.real_labels = []
        self.predicted_labels = []
        self.deep = 0
        count_max = 0
        self.lables = np.array(self.labels)
        self.features = np.array(self.features)
        for label in np.unique(self.labels):
            if self.labels.count(label) > count_max:
                count_max = self.labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if np.unique(self.lables).size < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    # split current node
    def split(self):
        # base conditions
        if self.num_cls == 1:
            self.splittable = False
            return self
        if self.features.size == 0:
            self.splittable = False
            return self

        branches = []
        entropy_of_node = 0
        total_label_count = len(self.labels)

        featr = self.features
        labl = self.labels

        for label in np.unique(labl):
            number = labl.count(label) / total_label_count
            if number > 0:
                entropy_of_node += -number * np.log2(number)

        arr = []
        for i in range(0, featr.shape[1]):
            attribute = featr[:, i]
            current_attribute_size_of_unique_values = np.unique(attribute).size
            current_feature_index = i

            for unique_attribute in np.unique(attribute):
                indices_array = np.where(featr[:, i] == unique_attribute)
                labels_arrray = []
                for index in indices_array[0]:
                    labels_arrray.append(labl[index])

                label_per_attribute_value = []
                label_counter = Counter(labels_arrray)
                for label in np.unique(labl):
                    if label in label_counter:
                        label_per_attribute_value.append(label_counter[label])
                    else:
                        label_per_attribute_value.append(0)

                branches.append(label_per_attribute_value)

            IG_attribute = Util.Information_Gain(entropy_of_node, branches)
            arr.append([IG_attribute, current_attribute_size_of_unique_values, current_feature_index])

            branches = []

        arr = sorted(arr, key=lambda  x:(x[0], x[1], -x[2]), reverse=True)

        self.dim_split = arr[0][2]
        unique_features = np.unique(featr[:, self.dim_split])

        for value in unique_features:
            indices_array = np.where(featr[:, self.dim_split] == value)

            labels_array_pass = []
            features_array_pass = []
            labels_array_left = []
            features_array_left = []

            for i, feature in enumerate(featr):
                if i in indices_array[0]:
                    features_array_pass.append(feature.tolist())
                    labels_array_pass.append(labl[i])
                else:
                    features_array_left.append(feature.tolist())
                    labels_array_left.append(labl[i])

            features_array_pass = np.delete(features_array_pass, self.dim_split, axis=1)

            featr = np.array(features_array_left)
            labl = np.array(labels_array_left)

            tree_node = TreeNode(features_array_pass, labels_array_pass, np.unique(labels_array_pass).size)
            tree_node.feature_uniq_split = value
            self.children.append(tree_node.split())

        return self

    # predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int

        if not self.splittable:
            return self.cls_max

        for children in self.children:
            if feature[self.dim_split] == children.feature_uniq_split:
                feature1 = np.delete(feature, self.dim_split, None)
                return children.predict(feature1)

        return self.cls_max

    def get_accuracy(self, X_test, y_test):
        prediction = []
        real = y_test
        accuracy = 0
        for idx, feature in enumerate(X_test):
            prediction.append(self.predict(feature))

        for i in range(len(prediction)):
            if prediction[i] == real[i]:
                accuracy += 1

        accuracy = accuracy/len(real)

        return accuracy

    def create_list(self, stack, queue):
        if queue:
            current = queue.popleft()
            stack.append(current)
            for children in current.children:
                queue.append(children)
            self.create_list(stack, queue)

        return stack

    def remove_children_from_stack(self, stack, child):
        if child.children == []:
            stack.remove(child)
            return
        for c1 in child.children:
            self.remove_children_from_stack(stack, c1)
        return

    def optimize_tree(self, X_test, y_test):
        queue = deque()
        queue.append(self)
        stack = []
        stack = self.create_list(stack, queue)

        current_accuracy = self.get_accuracy(X_test, y_test)
        current_node_address = 0

        flag = True

        while flag:
            flag = False
            for node in stack:
                if node.splittable:
                    deleted_children = node.children
                    node.splittable = False
                    node.children = []

                    acc = self.get_accuracy(X_test, y_test)

                    if acc > current_accuracy:
                        flag = True
                        current_node_address = node
                        current_accuracy = acc

                    node.splittable = True
                    node.children += deleted_children

            if flag:
                for child in current_node_address.children:
                    self.remove_children_from_stack(stack, child)
                current_node_address.splittable = False
                current_node_address.children = []
