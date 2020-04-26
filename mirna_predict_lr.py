#!/usr/bin/env python3
# Daniel B. Stribling
# Shaunak Sompura
# 2020-04-21
# Final Project for Machine Learning 
"""
This program is designed to implement a machine learning algorithm for the prediciton
of whether or not a miRNA will target a given candidate target sequence. 
First, training dataset is used with miRNA / target pairs to train the algorithm.
Then, these features are used to make predictions on a test dataset.
"""

import os
import sys
import numpy as np
import scipy as sp
import sklearn.model_selection
import sklearn.linear_model
import datetime
#import operator
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#setup program settings
use_modes = ['lr', 'ridge', 'lasso', 'elastic-net']
num_splits = 10
verbose = False

#setup program functions
_native_print = print
def flush_print(*args, **kwargs):
    _native_print(*args, '\\\\', **kwargs)
    sys.stdout.flush()

print = flush_print

def check_predictions(test_predictions, test_class, results, mode, verbose):
        correct = 0
        incorrect = 0
        with open(out_prefix + '%s_classification.tsv' % mode, 'a') as out_file_obj:
            out_file_obj.write('Split:%i\n' % split_i)
            out_file_obj.write('pred\tsign(pred)\ttrue_class\n')
            for i in range(len(test_predictions)):
                use_predict = np.sign(test_predictions[i])
                out_file_obj.write('%f\t%i\t%i\n' % (
                                                     test_predictions[i], 
                                                     use_predict, 
                                                     test_class[i]
                                                     ))
                #print(test_predictions[i], test_class[i])
                if use_predict == np.sign(test_class[i]):
                    correct += 1
                else:
                    incorrect += 1
        total = correct + incorrect
        correct_percent = (correct/total) * 100
        incorrect_percent = (incorrect/total) * 100
        if verbose:
            print('Results:')
            print('  Correct:  ', correct_percent)
            print('  Incorrect:', incorrect_percent)   
            print()

        results[mode]['percent_correct'].append(correct_percent)
        results[mode]['thetas'].append(clf.coef_)


# Begin Execution
start_time = datetime.datetime.now()
print('\nOutput for Final Project for Dan Stribling & Shaunak Sompura')

analysis_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(analysis_dir, 'source_data')
out_dir = os.path.join(analysis_dir, 'output_lr')
select_features_csv_path = os.path.join(analysis_dir, 'full_dataset_selected_features_v01.csv')
out_prefix = os.path.join(out_dir, 'output_')


# Reading in Features:
print ('Reading miRNA features:')

# Read Features here. Explanation of features found in: 
#  section 3.1 of https://academic.oup.com/bioinformatics/article/32/18/2768/1743913

with open(select_features_csv_path, 'r') as select_features_csv:
    mirna_data_header = next(select_features_csv).split(',')
    mirna_data_features = mirna_data_header[:]
    mirna_data_features[-1] = 'constant'
    max_header_len = max(len(h) for h in mirna_data_features)
mirna_data = np.genfromtxt(select_features_csv_path, delimiter=',', skip_header=1)
num_points, num_columns = mirna_data.shape

#change 1,0 classification to 1/-1 classification:
for i in range(num_points):
    if mirna_data[i,-1] == 0:
        mirna_data[i,-1] = -1


#print(mirna_data)
print('\nData Reading Complete.')
print('num_points x num_columns:', num_points, num_columns)
print()

shuffle_obj = sklearn.model_selection.ShuffleSplit(n_splits=num_splits, test_size=0.1)

#Clear old output files:
for mode in use_modes:
    with open(out_prefix + '%s_classification.tsv' % mode, 'w'):
        pass

#Prepare results files:
results = {mode:{'percent_correct':[], 'thetas':[]} for mode in use_modes}

print('\nBegining Testing, using shuffled train and test data using test fraction 0.1\n')

for split_i, (train_indices, test_indices) in enumerate(shuffle_obj.split(mirna_data), start=1):
    print('Training/Testing Split', split_i)
    
    num_train_points = train_indices.size
    num_test_points = test_indices.size
    
    shuffled_train = np.zeros((num_train_points, num_columns))
    shuffled_test = np.zeros((num_test_points, num_columns))
    
    for new_i, old_i in enumerate(train_indices):
        shuffled_train[new_i, :] = mirna_data[old_i, :]
    
    for new_i, old_i in enumerate(test_indices):
        shuffled_test[new_i, :] = mirna_data[old_i, :]
    
    #print(mirna_data[1,:])
    #print(shuffled_train[1,:])
    #print(shuffled_test[1,:])
    
    train_features = shuffled_train[:, :]
    train_class = np.copy(shuffled_train[:, -1])
    test_features = shuffled_test[:, :]
    test_class = np.copy(shuffled_test[:,-1])

    # Set final feature as a constant term 1
    for i in range(len(train_features)):
        train_features[i, -1] = 1 
    for i in range(len(test_features)):
        test_features[i, -1] = 1


    if verbose:
        print('\nTraining and Test Divisions Complete.')
        print('  Training: num_points x num_features:\n', train_features.shape)
        print('  Testing:  num_points x num_features:\n', test_features.shape)
        print()
    
    #Begin Linear Regression
    if 'lr' in use_modes:
        mode = 'lr'
        if verbose:
            print('Performing Linear Regression of Training Data using sklearn')
        clf = sklearn.linear_model.LinearRegression()

        if verbose:
            print('Training Model...')
        clf.fit(train_features, train_class)

        if verbose:
            print('Making Predictions...')
        test_predictions = clf.predict(test_features)
        
        if verbose:
            print()
    
    check_predictions(test_predictions, test_class, results, mode, verbose)

    #Begin Linear Regression
    if 'ridge' in use_modes:
        mode = 'ridge'
        if verbose:
            print('Performing Linear Regression of Training Data using sklearn')
        clf = sklearn.linear_model.Ridge()

        if verbose:
            print('Training Model...')
        clf.fit(train_features, train_class)

        if verbose:
            print('Making Predictions...')
        test_predictions = clf.predict(test_features)
        
        if verbose:
            print()
    
    check_predictions(test_predictions, test_class, results, mode, verbose)

    #Begin Linear Regression
    if 'lasso' in use_modes:
        mode = 'lasso'
        if verbose:
            print('Performing Linear Regression of Training Data using sklearn')
        clf = sklearn.linear_model.Lasso(alpha=0.1)

        if verbose:
            print('Training Model...')
        clf.fit(train_features, train_class)

        if verbose:
            print('Making Predictions...')
        test_predictions = clf.predict(test_features)
        
        if verbose:
            print()
    
    check_predictions(test_predictions, test_class, results, mode, verbose)

    #Begin Linear Regression
    if 'elastic-net' in use_modes:
        mode = 'elastic-net'
        if verbose:
            print('Performing Linear Regression of Training Data using sklearn')
        clf = sklearn.linear_model.ElasticNet(alpha=0.1)

        if verbose:
            print('Training Model...')
        clf.fit(train_features, train_class)

        if verbose:
            print('Making Predictions...')
        test_predictions = clf.predict(test_features)
        
        if verbose:
            print()
    
    check_predictions(test_predictions, test_class, results, mode, verbose)

print()
for mode in use_modes:
    mode_results = results[mode]
    results_str = ''
    results_str += 'Results for mode: %s\n' % mode
    av_correct =  np.average(mode_results['percent_correct'])
    results_str += '  Av. Percent Correct: %.2f%%\n' % av_correct
    #print('  Thetas:')
    #for theta in mode_results['thetas']:
    #    print(' ', theta)
    results_str += '  Av. weights:\n'
    av_thetas = np.average(mode_results['thetas'], axis=0)
    for i in range(len(mirna_data_features)):
        results_str += '    ' 
        results_str += (mirna_data_features[i] + ':').ljust(max_header_len+1)
        results_str += '%s' % (' ' * int(av_thetas[i] >= 0))
        results_str += '%.5f' % av_thetas[i]
        results_str += '\n'

    print(results_str.rstrip())
    with open(out_prefix + '%s_results.txt' % mode, 'w') as out_file_obj:
        out_file_obj.write(results_str)

    print('  Plotting.')
    names = ['div_%i' % i for i in range(1, num_splits + 1)]
    plt.clf()
    plt.bar(names, mode_results['percent_correct'])
    plt.xlabel('Data Division Iteration')
    plt.ylabel('%-Correct')
    plt.ylim(0,100)
    plt.title('%s : Av. Percent correct (n=%i)' % (mode, num_splits))
    out_name = out_prefix + '%s_percent_plot' % mode
    plt.savefig(out_name)

    plt.clf()
    for theta in mode_results['thetas']:
        plt.plot(theta)
    plt.xlabel('miRNA Feature:')
    plt.ylabel('Av. Feature Weight (n=%i)' % num_splits)
    plt.legend(names)
    plt.title('%s : Feature Weights' % mode)
    out_name = out_prefix + '%s_feature_weights_plot' % mode
    plt.savefig(out_name)

    print()
print('\nDone.\n')

