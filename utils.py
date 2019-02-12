import numpy as np


def Information_Gain(S, branches):
    branches = np.array(branches)
    total_number_of_values = branches.sum()
    
    averaged_entropy = 0
    attribute_entropy = 0
    for branch in branches:
        sum_branch = branch.sum()
        for num_classes in branch:
            number = num_classes/sum_branch
            if number > 0:          
                attribute_entropy += number*np.log2(number)
        averaged_entropy += -(sum_branch/total_number_of_values)*attribute_entropy
        attribute_entropy = 0

    return float(format(S - averaged_entropy, '.10f'))


def reduced_error_prunning(decisionTree, X_test, y_test):
    decisionTree.add_real_predicted_label_property(X_test, y_test)


def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


