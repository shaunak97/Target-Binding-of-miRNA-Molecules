#!/usr/bin/env python3
# Daniel B. Stribling
# Shaunak Sompura
# 2020-04-21
# Final Project for Machine Learning 
"""
This script selects a subset of features of the raw features file for input into
the subsequent script.
"""

import os
import sys
import csv

_native_print = print
def flush_print(*args, **kwargs):
    _native_print(*args, **kwargs)
    sys.stdout.flush()

# Set script directories and input file names.
analysis_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(analysis_dir, 'source_data')
raw_features_csv_path = os.path.join(data_dir, 'full_dataset_v01.csv')
out_dir = analysis_dir
select_features_csv_path = os.path.join(out_dir, 'full_dataset_selected_features_v01.csv')

keys_printed = False

use_keys = {
            'fold_energy',
            'seed_match_count',
            'seed_match_full',
            'mirna_au_count',
            'mirna_au_frac',
            'count_matches',
            'frac_matches',
            'len_target_reg',
            'longest_pairing',
            '3p_mirna_match_count',
            'seed_3p_diff',
           }

# Select features from all possible input csv features:
with open(raw_features_csv_path, 'r', newline='') as raw_features_csv, \
     open(select_features_csv_path, 'w', newline='') as select_features_csv:

    csv_dict_reader = csv.DictReader(raw_features_csv)
    csv_writer = csv.writer(select_features_csv)

    for line_dict in csv_dict_reader:
        if not keys_printed:
            print('All Keys:')
            for key in line_dict.keys():
                print('  -', key)
            print()
            print('Using Keys:')
            header = []
            for key in line_dict.keys():
                if key in use_keys:
                    print('  -', key)
                    header.append(key)

            # use "positive" key (prediction result) but place at end.
            print('  - positive')
            header.append('positive')

            print()
            csv_writer.writerow(header)
            keys_printed = True

        write_items = []           
        for key in line_dict.keys():
            if key in use_keys:
                write_items.append(line_dict[key])
        write_items.append(line_dict['positive'])
        csv_writer.writerow(write_items)
 
print('Done.')
print()
sys.exit()

 
