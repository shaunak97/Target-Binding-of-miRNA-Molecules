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
import random
from sklearn.feature_extraction.text import TfidfTransformer
import datetime
#import operator
#from matplotlib import pyplot as plt

_native_print = print
def flush_print(*args, **kwargs):
    _native_print(*args, **kwargs)
    sys.stdout.flush()

print = flush_print
start_time = datetime.datetime.now()

print('\nOutput for Final Project for Dan Stribling & Shaunak Sompura)')

analysis_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(analysis_dir, 'source_data')
out_dir = analysis_dir
select_features_csv_path = os.path.join(out_dir, 'CLASH_Helwak_Selected_features.csv')


# Reading in Features:

# Read Features here. Explanation of features found in: 
#  section 3.1 of https://academic.oup.com/bioinformatics/article/32/18/2768/1743913

with open(select_features_path, 'r') as select_features:
    #read into numpy array
    #then do ml.




# Legacy code from previous homework problem begins below, left to potentially reuse elements such as plotting.



num_words = len(vocabulary)
print('Indices for %i words read.\n' % num_words)

print('Reading Map File...')
indices_from_number = {}
indices_from_category = {}
with open(map_file, 'r') as map_file_obj:
    for line in map_file_obj:
        line_split = line.split()
        #print(line_split)
        indices_from_number[int(line_split[1])] = line_split[0]
        indices_from_category[line_split[0]] = int(line_split[1])
print('Descriptions for %i categories read.\n' % len(indices_from_number))

#Data has form: docIdx wordIdx count
#17500
#print(indices_from_number)
#print(indices_from_category)

print('Reading Training Classifications into array.')
train_y = np.loadtxt(train_label_file, dtype=int)
num_train_docs = len(train_y)

print('Reading Training Features into %i x %i matrix' % (num_train_docs, num_words))
train_features = np.zeros((num_train_docs, num_words))
with open(train_data_file, 'r') as train_data_file_obj:
    for line in train_data_file_obj:
        line_split = line.strip().split()
        doc = int(line_split[0])
        doc_i = doc - 1
        word = int(line_split[1])
        word_i = word - 1
        count = int(line_split[2])
        train_features[doc_i,word_i] = count
print('Done Reading.')
train_features = sp.sparse.csr_matrix(train_features)

#print('Reading Test Classifications into array.')
#test_y = np.loadtxt(test_label_file, dtype=int)
#num_test_docs = len(test_y)

#print('Reading Test Features into %i x %i matrix' % (num_test_docs, num_words))
#test_features = np.zeros((num_test_docs, num_words))
#with open(test_data_file, 'r') as test_data_file_obj:
#    for line in test_data_file_obj:
#        line_split = line.strip().split()
#        doc = int(line_split[0])
#        doc_i = doc - 1
#        word = int(line_split[1])
#        word_i = word - 1
#        count = int(line_split[2])
#        test_features[doc_i,word_i] = count

print('Transforming features to Token-Frequency * Inverse-Document-Frequency '
      + '(TF-IDF) representations:')

transformer = TfidfTransformer()
print('Fitting TF-IDF Features...')
transformer.fit(train_features)
print(transformer.idf_)
print('Creating TF-IDF representation for training features...')
tfidf_train_features = transformer.transform(train_features)
tfidf_train_features_array = sp.sparse.csr_matrix(tfidf_train_features.toarray())

# print(tfidf_features_array, '\n')
# print(type(tfidf_features_array), '\n')
# print(tfidf_features_array[0,:], '\n')
# print(tfidf_features_array[0,:15], '\n')
# print(tfidf_features_array[1,:15], '\n')

#print('Creating TF-IDF representation for test features...')
#tfidf_test_features = transformer.transform(test_features)
#tfidf_test_features_array = tfidf_test_features.toarray()

print('Reading of Input Files Complete.\n')

print(np.shape(tfidf_train_features_array))
#print(np.shape(tfidf_test_features_array))
print(np.shape(train_y))

num_documents, num_words = np.shape(train_features)
print('Num Documents:', num_documents)
print('Num Words:', num_words)

phi_t = tfidf_train_features_array
phi = phi_t.transpose(copy=True) 
use_m, use_n, = np.shape(phi)
#num_test_features, _ = np.shape(test_features)

if DO_ORTHOGONAL:
    start_time = datetime.datetime.now()
    print('Beginning PCA Analysis via Orthogonal Iteration')
    k = 2
    total_iterations = 10000
    store_thetas = {i:None for i in [1, 10, 50, 100]}
    store_theta_diffs = {}

    rand_init = np.random.rand(use_m, k)
    theta, _ = np.linalg.qr(rand_init) 

    print('\nTheta Init:')
    print(theta)
    print(np.shape(theta), '\n')
    #input()

    for i in range(1,total_iterations + 1):
        last_theta = theta
        phi_t_theta = phi_t.dot(theta)
        theta_tilde = phi.dot(phi_t_theta)
        theta, _ = np.linalg.qr(theta_tilde)
        theta_diff = np.linalg.norm(last_theta - theta)

        #if (i % 10) == 0 or i in store_thetas:
        #    print(i)

        if i in store_thetas:
            store_thetas[i] = theta
            store_theta_diffs[i] = theta_diff
        print(i, theta_diff)
        
        if theta_diff < (10**(-15)):
            store_thetas[i] = theta
            store_theta_diffs[i] = theta_diff
            break

    print()

    for i_num in store_thetas.keys():
        print('Outputting Results for point:', i_num)
        theta_i = store_thetas[i_num]
        theta_diff_i = store_theta_diffs[i_num]
        phi_dense = phi.todense()
        y_matrix = np.dot(np.transpose(theta_i), phi_dense)    

        #print('  Dividing into categories')
        category_sums = {i:0 for i in range(1, 21)}
        for i in range(num_documents):
            category_sums[int(train_y[i])] += 1
        #print(sum(category_sums.values()))
     
        points_by_category = {}

        for i in range(num_documents):
            #if (i % 100) == 0:
            #    print(i)
            c = int(train_y[i])
            if c not in points_by_category:
                points_by_category[c] = np.column_stack([y_matrix[:,i]])
            else:
                points_by_category[c] = np.append(points_by_category[c],
                                                  np.reshape(y_matrix[:,i], (k,1)),
                                                  axis=1)

        colors = [('tab:%s' % c) for c in ['blue', 'orange', 'green', 'red', 'purple',
                                           'brown', 'pink', 'gray', 'olive', 'cyan']]
        colors += ['black', 'lime', 'slateblue', 'darkgoldenrod', 'darkslategray',
                   'yellow', 'orangered', 'crimson', 'aquamarine', 'palegreen']

        print(len(colors))

        print('  Plotting.')
        plt.clf()
        for i, i_points in points_by_category.items():
            plt.plot(i_points[0,:], i_points[1,:], color=colors[i-1], marker='.')
        plt.xlabel('Axis_1')
        plt.ylabel('Axis_2')
        plt.title('Stribling HW4: Orthogonal Iteration: %i Iterations' % i_num)
    
        out_name = 'Plot_Orthogonal_Iterations_%i' % i_num
        plt.savefig(out_name)

    print('Part A time:', datetime.datetime.now() - start_time, '\n')

if DO_EM:
    start_time = datetime.datetime.now()
    print('Beginning PCA Analysis via Orthogonal Iteration')
    k_pca = 100
    total_iterations_pca = 10000
    break_at = (0.5* 10**(-2))

    k_gmm = 20
    total_iterations_gmm = 150

    rand_init = np.random.rand(use_m, k_pca)
    theta, _ = np.linalg.qr(rand_init) 

    print('\nTheta Init:')
    print(theta)
    print(np.shape(theta), '\n')
    #input()

    # PCA Component
    for i in range(1,total_iterations_pca + 1):
        last_theta = theta
        phi_t_theta = phi_t.dot(theta)
        theta_tilde = phi.dot(phi_t_theta)
        theta, _ = np.linalg.qr(theta_tilde)
        theta_diff = np.linalg.norm(last_theta - theta)

        print(i, theta_diff)
        
        if theta_diff < break_at:
            break

    y_matrix = np.dot(np.transpose(theta), phi.todense())

    print('\nPCA Component Complete.')
    print('Phi Shape:', np.shape(phi))
    print('Phi^T Shape:', np.shape(phi_t))
    print('Theta Shape:', np.shape(theta))
    print('Y Shape:', np.shape(y_matrix))

    # GMM Component:
    print('\nBeginning GMM Component.')

    gmm_theta = theta
    gmm_phi = y_matrix
    num_features, num_points = np.shape(gmm_phi)
    clusters = range(1, (k_gmm + 1))

    #Initialze with equal cluster distribution: pi = 1/k
    i_pi = {i: (1.0/k_gmm) for i in clusters}

    #Initialize with random slection of cluster center:
    init_indexes = {i: np.random.randint(num_points) for i in clusters}
    i_mu = {i: gmm_phi[:,init_indexes[i]] for i in clusters}
    
    #Initialize sigma with the identity matrix:
    #add_smidgeon = np.ones(num_features) * (10**(-30))
    i_sigma = {i: (np.identity(num_features)) for i in clusters}

    print('GMM Phi Shape:', np.shape(gmm_phi))
    print('GMM Theta Shape:', np.shape(gmm_theta))
    print('GMM Mu Shape:', np.shape(i_mu[1]))
    print('GMM Sigma Shape:', np.shape(i_sigma[1]))

    print('Beginning Expectation/Maximization') 
    for iteration in range(1,total_iterations_gmm + 1):
        # Expectation
        psi = np.zeros((num_points, k_gmm)) 
        # Prepare for computations:
        mvn_f = {i: None for i in clusters}
        for c in clusters:
            mu_c = np.ravel(i_mu[c])
            sigma_c = i_sigma[c]
            mvn_f[c] = sp.stats.multivariate_normal(mu_c, sigma_c)

        for i in range(num_points):
            x_i = np.ravel(gmm_phi[:,i])

            i_denominator = 0
            i_denominator = sum((i_pi[c] * mvn_f[c].pdf(x_i) for c in clusters))

            for c in clusters:
                psi[i, (c-1)] = ( i_pi[c] * mvn_f[c].pdf(x_i) / i_denominator )                

        #Maximization
        #last_i_pi = deepcopy(i_pi)
        last_i_mu = deepcopy(i_mu)
        #last_i_sigma = deepcopy(i_sigma)     
        for c in clusters:
            c_index = c - 1
            sum_pi_ic = np.sum(psi[:, c_index])
            i_pi[c] = sum_pi_ic / num_points
            i_mu[c] = ( 1 / sum_pi_ic ) * np.dot(gmm_phi, psi[:, c_index])
            
            i_sigma_c_sum = np.zeros_like(i_sigma[c])
            for i in range(num_points):
                phi_demeaned_i = np.ravel(gmm_phi[:,i]) - np.ravel(i_mu[c])
                i_sigma_c_sum += (
                                  psi[i, c_index] 
                                  * np.outer(phi_demeaned_i, phi_demeaned_i)
                                 )
            i_sigma[c] = (1 / sum_pi_ic) * i_sigma_c_sum 

        all_old_mu = np.ravel(np.concatenate([val for val in last_i_mu.values()]))
        all_new_mu = np.ravel(np.concatenate([val for val in i_mu.values()]))
        iter_distance = np.linalg.norm(all_old_mu - all_new_mu)
        print(iteration, iter_distance)

    c_wordmaps = {c: [] for c in clusters}

    for c in clusters:
        print('Cluster:', c)
        c_weights = np.ravel(np.dot(gmm_theta, np.reshape(i_mu[c], (num_features,1))))
        c_most_important = {}
        for (sort_pos, (orig_idx, orig_val)) in enumerate([(idx,val) for (idx,val) 
             in reversed(sorted(enumerate(c_weights), key=lambda x: x[1]))]):
             word = vocabulary[orig_idx + 1]
             print(sort_pos, orig_idx, orig_val, word)
             c_wordmaps[c].append(word)
             if sort_pos == 9:
                 break

    with open('out_cluster_results.csv', 'w') as cluster_csv:
        cluster_csv.write(','.join(['cluster'] + ['word%i' % i for i in range(1, 11)] + '\n'))
        for c, c_items in c_wordmaps.items():
            cluster_csv.write(','.join([str(c)] + c_items) + '\n')

print()
sys.exit()

 
