""" This module contains some utility procedures for classifier
Main algorithm: you need to create Classifier with desired parameters (build_classifier.__doc__),
choose targets of CSV-file (choose_targets.__doc__),
pre-process CSV-file with features and targets (read_data_set.__doc__),
Classifier.fit: build classification tree with pre-processed data,
Classifier.predict
"""
import StringIO
import argparse
import numpy
import scipy.stats
import sklearn.feature_extraction
import pydot
from myTree import MyClassifier
from myTree import BinaryTree

TARGET = "target"


def choose_targets(source):
    """
    0 = VK - CSV - file
    1 = DT Tutorial (Seminar task)
    :return: chosen columns for classification
    """
    if source == 0:
        return 'vk'
    elif source == 1:
        columns = {
            "age": (4, parse_int),
            "gender": (5, parse_int),
            "education": (29, parse_string),
            TARGET: (54, parse_int)
        }
        return columns


def read_data_set(ds_path, columns, with_targets=True):
    """ Parse CSV-file, choose no-None data

    :param ds_path: CSV-file path
    :param columns: returned from choose_targets
    :return: x(n x m), y(n x 1), feature_names
    """
    data = []
    targets = []
    with open(ds_path, 'rU') as ds_file:
        for line in ds_file:
            items = line.strip().split('\t')

            row = {}
            target = None
            row_valid = True
            for name, (column, parse_fun) in columns.iteritems():
                value = parse_fun(items[int(column) - 1])
                if value is None:
                    row_valid = False
                    break
                if name == TARGET:
                    target = value
                else:
                    row[name] = value

            if with_targets:
                if row_valid and row and target:
                    data.append(row)
                    targets.append(target)
            else:
                # For clf.predict()
                if row_valid and row:
                    data.append(row)

    dv = sklearn.feature_extraction.DictVectorizer()
    return dv.fit_transform(data).todense(), numpy.array(targets), dv.get_feature_names()


def get_data_for_fit_and_cv(columns, ds_path='data/epoll.csv'):
    """

    :param columns: Get it from trees.choose_targets(1)
    :param ds_path: (DEFAULT: 'data/epoll.csv')
    :return: x, y - numpy.arrays for fitting tree, cv_x, cv_y - for cross-validation in pruning
    """
    x, y, names = read_data_set(ds_path, columns)
    l = len(y)
    return x[0:l*8/10], y[0:l*8/10], x[l*8/10:l], y[l*8/10:l]


def print_set_stats(ds, target, feature_names):
    print "Data set contains {} items and {} features".format(ds.shape[0], ds.shape[1])

    def print_distribution(x):
        for value, count in scipy.stats.itemfreq(x):
            print "{value}\t{count}".format(value=value, count=count)

    for i, name in enumerate(feature_names):
        print "Feature: {}".format(name)
        print_distribution(ds[:, i])

    print "Target"
    print_distribution(target)


def fit_decision_tree(x, y):
    #model = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1000, max_depth=2)

    #scores = cross_val_score(model, x, y, cv=20)
    #print "Model mean accuracy: {}".format(numpy.mean(scores))

    model = MyClassifier().fit(x, y)

    return model


def build_classifier(min_delta_imp=500, min_list_size=200, max_tree_node_amount=100500):
    """ Build classifier
    :param min_delta_imp: (=100) split-stopping criteria
    :param min_list_size: (=10) -||-
    :return: Classifier
    """
    return MyClassifier(min_delta_imp, min_list_size, max_tree_node_amount)


#def export_model(model, out_path):
#    dot_data = StringIO.StringIO()
#    tree.export_graphviz(model, out_file=dot_data)
#    graph = pydot.graph_from_dot_data(dot_data.getvalue())
#    graph.write_pdf(out_path)


def parse_int(s):
    return int(s) if s != "-1" else None


def parse_string(s):
    return s if s != "-1" else None


def main():
    print "## Welcome to the decision trees tutorial ##"
    args = parse_args()

    print "parsed"

    #Seminar tutorial
    columns = choose_targets(1)

    x, y, cv_x, cv_y = get_data_for_fit_and_cv(columns)

    #print_set_stats(x, y, feature_names)

    cls = build_classifier(500, 200, 100)
    model = cls.fit(x, y)

    cls.pruning(cv_x, cv_y, 2)

#    if args.out_path:
#        export_model(model, args.out_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Experiments with decision trees')
    parser.add_argument('-o', dest='out_path', help='a path to the exported tree')
    parser.add_argument('ds_path', nargs=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()