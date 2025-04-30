import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import csv
import sys
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import cleaners as cln
import numpy as np
import argparse
import pandas as pd

from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import Dense, Concatenate
from datasets import get_tf_database, load_regression, load_regression_dirty, load_features_and_data, reverse_categorical_columns, reverse_to_input_domain, get_date_columns
from latent_operators import LatentOperator
from transformation_in_x import apply_transformation_in_x, include_errors_at_random
from utils import create_and_train_LOP, create_and_train_classifier
from error_detection import predict_on_enhanced, eval_correctly, eval_numeric_rmse
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
from copy import deepcopy

parser = argparse.ArgumentParser(description="Generalizable Data Cleaning of Tabular Data in Latent Space")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--latent", type=int, default=240)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--dataset", default='adult')
parser.add_argument("--experiment", default='vs_dirty')
parser.add_argument("--eval_tuples", type=int, default=-1)

args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)
print("Is Eager execution?",tf.executing_eagerly())
LOP = None

@tf.function
def _translate_all_columns_by_1(inputs):
    zs  = inputs
    for column in range(zs.shape[0]):
        new_z = LOP.translate_operator(zs[column], 1)
        zs = tf.tensor_scatter_nd_update(zs, [column], new_z)
    return zs

def generate_cleaned_data(x_, Zs, Ks, decoder):
    #avoid decoding Zs when there is no error, just use the value
    xs = tf.squeeze(tf.transpose(decoder(tf.unstack(Zs, axis = 1)), [1,0,2]))
    x_p = tf.where(Ks == 0, x_, xs) 
    return x_p


#CONFIGS========================================================
ds = args.dataset
T_EXAMPLES = args.eval_tuples
V_EXAMPLES = args.eval_tuples 
T_EVAL_EXAMPLES = args.eval_tuples
V_EVAL_EXAMPLES = args.eval_tuples
MODEL_EPOCHS = 200
K_EPOCHS = args.epochs
MISSING_REPLACE = '3.0'
T = 'missing_values'
K = args.K

#LOAD AIRBNB, NASA or BIKE=====================================
start = time.time()

#MAX AND MIN FOR THE ALREADY FILTERED SET + TARGET
x_clean_train, y_clean_train, x_clean, y_clean, MAX, MIN, SCALER, CAT_ENCODER = load_regression(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, True, normalize_sklearn = True)
x_dirty_train, y_dirty_train, x_test, y_test = load_regression_dirty(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)

print(y_test.shape)
COLS = x_clean_train.shape[1]
print("# COLUMNS:", COLS)

train_dataset = get_tf_database(x_clean_train[0:T_EXAMPLES],
                                x_clean_train[0:T_EXAMPLES],
                                args.batch_size)

val_dataset = get_tf_database(x_clean[0:V_EXAMPLES],
                              x_clean[0:V_EXAMPLES],
                              args.batch_size)
print(time.time() - start)


# TRAIN MODELS========================================================
start = time.time()
encoder, decoder, LOP, t_acc, v_acc = create_and_train_LOP(train_dataset,
                                                           val_dataset,
                                                           COLS,
                                                           args.latent,
                                                           K,
                                                           1,
                                                           T,
                                                           epochs=args.epochs,
                                                           model_name=args.dataset)
print("LOP ",time.time() - start, "sec", "Train loss:", t_acc," VAL loss:" , v_acc)

for el in encoder.layers:
     el.trainable = False
for el in decoder.layers:
    el.trainable = False

   
#Default Classifier MODEL========================================================================
t_dataset = get_tf_database(x_clean_train, y_clean_train, args.batch_size)
v_dataset = get_tf_database(x_clean, y_clean, args.batch_size)    

start = time.time()
nnreg_model = create_and_train_classifier(t_dataset,
                                          v_dataset,
                                          COLS,
                                          args.latent,
                                          T,
                                          n_epochs=MODEL_EPOCHS,
                                          model_name=args.dataset)
print("NN train ", time.time() - start, "sec")


#Enhanced MODEL===============================================================================
concatenation_model = Concatenate()(decoder.output)
enhanced_model = Model(inputs=decoder.input, outputs= nnreg_model(concatenation_model))
enhanced_model.compile(optimizer='adam',loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

n_scores = []
en_scores = []
en_smart_scores = []
clean_scores = []

#MISSING P percentage of values==================================================
x_test = x_test[0:V_EXAMPLES] 
y_test = y_test[0:V_EXAMPLES]
print("# tuples for test:", x_test.shape[0])

#percentages_of_noise = [0.0]
# clean_scores.append(nnreg_model.evaluate(x_clean, y_clean)[1])
# n_scores.append(nnreg_model.evaluate(x_test, y_test)[1]) #dirty score
# Zs, Ks = predict_on_enhanced(x_test, LOP, encoder, decoder, _translate_all_columns_by_1)
# en_scores.append(enhanced_model.evaluate(tf.unstack(Zs, axis = 1), y_test)[1])
# detections = eval_correctly(x_test, y_test, Zs, Ks, decoder, nnreg_model)
# en_smart_scores.append(detections)
# #eval_correctly(x_test, y_test, Zs, Ks, decoder, nnreg_model)



############ WRITE CLEAN DATA #################
headers,  target_name, dirty_data, _, data_with_y, _, clean_data, FULL_SCALER, CAT_ENCODER = load_features_and_data(ds, T_EVAL_EXAMPLES, V_EVAL_EXAMPLES, MISSING_REPLACE, SCALER, CAT_ENCODER, True, MAX, MIN, normalize_sklearn = True)
full_header = headers["full_header"]
header_with_y = headers["filtered_header_with_y"]
filtered_header = headers["filtered_header"]
numeric_header = headers["numeric_header"] 
categorical_header = headers["categorical_header"]

X_dirty = deepcopy(data_with_y[filtered_header]).to_numpy()
Zs_csv, Ks_csv = predict_on_enhanced(X_dirty, LOP, encoder, decoder, _translate_all_columns_by_1)
clean_csv = generate_cleaned_data(X_dirty, Zs_csv, Ks_csv, decoder)


#GENERATE THE RMSE COMPARISION TO REIN#######################################
rmse_dirty = mean_squared_error(clean_data[numeric_header], dirty_data[numeric_header], squared = False)

#EVAL NUMERIC RMSE###########################################################
df_cleaned = deepcopy(dirty_data)
df_cleaned[filtered_header] = clean_csv
rmse = mean_squared_error(clean_data[numeric_header], df_cleaned[numeric_header], squared = False)

def _clean_with_error_detector(_df_cleaned, _dirty_data, _detections):
    rows_ = _detections[:, 0] - 2 # rein index starts at 1 and jumps the header
    cols_ = _detections[:, 1]
    df_ = deepcopy(_dirty_data) #clean_data)
    #only use know dirty position repairs
    for r, c in zip(rows_, cols_):
        df_.iloc[r, c] =  _df_cleaned.iloc[r, c]
    return df_

rmse_ed2 = 0
rmse_dboost = 0
rmse_gt = 0

#EVAL NUMERIC RMSE BY COLUMN################################################
rmse_by_column_dirty = 0
rmse_by_column_clean = 0
rmse_by_column_gt = 0
n_num_cols = len(numeric_header) #len(numeric_header) + 1 #rein benchmark includes the target with 0 errors
if n_num_cols > 0 :
    print("\n  RMSE (less is better) on Numerical Columns Dirty vs Clean")
    for num_col in numeric_header:
        r_clean = mean_squared_error(clean_data[num_col], df_cleaned[num_col], squared = False)
        r_dirty = mean_squared_error(clean_data[num_col], dirty_data[num_col], squared = False)
        #r_gt = mean_squared_error(clean_data[num_col], df_gt_cleaned[num_col], squared = False)
        print(f'{num_col}', r_dirty, " vs ", r_clean)
        rmse_by_column_dirty += r_dirty
        rmse_by_column_clean += r_clean
        #rmse_by_column_gt += r_gt

    rmse_by_column_clean =  rmse_by_column_clean / n_num_cols
    rmse_by_column_dirty =  rmse_by_column_dirty / n_num_cols
    #rmse_by_column_gt =  rmse_by_column_gt / n_num_cols

    print(f'TOTAL: {rmse_by_column_dirty} vs {rmse_by_column_clean}')#  vs {rmse_by_column_gt}')



#EVAL CATEGORICAL ACCURACY BY COLUMN###################################################
c = reverse_to_input_domain(args.dataset, clean_data, FULL_SCALER, CAT_ENCODER)
d = reverse_to_input_domain(args.dataset, dirty_data, FULL_SCALER, CAT_ENCODER)
l = reverse_to_input_domain(args.dataset, df_cleaned, FULL_SCALER, CAT_ENCODER)
pd.options.display.max_columns = None
pd.options.display.max_rows = None



precision_categories_dirty = 0
precision_categories = 0
recall_categories_dirty = 0
recall_categories = 0
accuracy_categories_dirty = 0
accuracy_categories = 0
accuracy_categories_gt_detections = 0
accuracy_categories_dboost_detections = 0
accuracy_categories_ed2_detections = 0
n_cat_cols = len(categorical_header) #- 1 #beer_name




def _cat_metric(metric, clean, dirty, repaired):
    avg_type = "weighted"    
    #dirty, clean
    return metric(clean, dirty, average = avg_type, zero_division = 0.0), metric(clean, repaired, average = avg_type, zero_division = 0.0) 



if n_cat_cols > 0:
    print("\n  F1 (more is better) on Categorical Columns Dirty vs Clean")
    for cat_col in categorical_header:
        f1_dirty, f1_clean = _cat_metric(f1_score, c[cat_col], d[cat_col], l[cat_col])
        pre_dirty, pre_clean = _cat_metric(precision_score, c[cat_col], d[cat_col], l[cat_col])
        rec_dirty, rec_clean = _cat_metric(recall_score, c[cat_col], d[cat_col], l[cat_col])

        accuracy_categories_dirty += f1_dirty
        precision_categories_dirty += pre_dirty
        recall_categories_dirty += rec_dirty

        accuracy_categories += f1_clean        
        precision_categories += pre_clean
        recall_categories += rec_clean

        print(f'F1 {cat_col}', f1_dirty, " vs ", f1_clean)
        print(f'Precision {cat_col}', pre_dirty, " vs ", pre_clean)
        print(f'Recall {cat_col}', rec_dirty, " vs ", rec_clean)

    accuracy_categories_dirty =  accuracy_categories_dirty / n_cat_cols
    accuracy_categories =  accuracy_categories / n_cat_cols
    precision_categories_dirty =  precision_categories_dirty / n_cat_cols
    precision_categories =  precision_categories / n_cat_cols
    recall_categories_dirty =  recall_categories_dirty / n_cat_cols
    recall_categories =  recall_categories / n_cat_cols
    
    accuracy_categories_dboost_detections =  accuracy_categories_dboost_detections / n_cat_cols
    accuracy_categories_ed2_detections =  accuracy_categories_ed2_detections / n_cat_cols
    accuracy_categories_gt_detections =  accuracy_categories_gt_detections / n_cat_cols

    print(f'TOTAL F1: {accuracy_categories_dirty} vs  {accuracy_categories}')

dirty_data[filtered_header] = df_cleaned[filtered_header]#clean_csv
lop_data = reverse_to_input_domain(args.dataset, dirty_data, FULL_SCALER, CAT_ENCODER)

#replace the date columns
lop_data = get_date_columns(ds, lop_data, dirty_data)

lop_data.to_csv(f'./DATASETS_REIN/{args.dataset}/LOP.csv', index = False)



###EVALUATE REIN################################################################################
rmse_dirty, rmse_repaired = cln.evaluate(f"./DATASETS_REIN/{args.dataset}/clean.csv",
                                         f"./DATASETS_REIN/{args.dataset}/dirty01.csv",
                                         f"./DATASETS_REIN/{args.dataset}/LOP.csv",
                                         args.eval_tuples)

print("Dirty vs LOP on REIN BENCHMARK: ", rmse_dirty, rmse_repaired)

#save results for plots
for_plots = pd.DataFrame({'rmse_numeric':rmse_repaired, 'rmse_gt_detector':rmse_gt, 'rmse_dirty': rmse_dirty,
                          'rmse_dboost': rmse_dboost, 'rmse_ed2': rmse_ed2,
                          'accuracy_categorical': accuracy_categories,
                          'accuracy_categorical_dirty': accuracy_categories_dirty,
                          'precision_categorical': precision_categories,
                          'precision_categorical_dirty': precision_categories_dirty,
                          'recall_categorical': recall_categories,
                          'recall_categorical_dirty': recall_categories_dirty,
                          'accuracy_categorical_ed2_detections': accuracy_categories_ed2_detections,
                          'accuracy_categorical_dboost_detections': accuracy_categories_dboost_detections,
                          'accuracy_categorical_gt_detections': accuracy_categories_gt_detections}, index = [0])
for_plots.index.name = "rein"

if not os.path.isdir("./evaluation"):
    os.makedirs("./evaluation")

for_plots.to_csv(f'./evaluation/rein_rmse_{args.dataset}.csv')

########### WRITE DETECTION DICTIONARY ##########
#get row and column where there is no error (i.e. k = 0 = identity or equal to K)
detection_dictionary_indexes = tf.where(tf.math.logical_or(tf.equal(Ks_csv, 0), tf.equal(Ks_csv, K)))
det_idx = detection_dictionary_indexes.numpy()


#ORDER By COLUMN INDEX AND REPLACE IT WITH CORRECT ID OF THE FULL DATASET
replacement_indexes = [full_header.index(i) for i in filtered_header if i in  full_header]
#print(full_header, filtered_header, replacement_indexes)

det_idx = pd.DataFrame(det_idx).sort_values([1, 0])

for idx, rplc in enumerate(replacement_indexes):
    det_idx = det_idx.replace(str(idx), rplc)


#FIX FOR REIN BENCHMARK INDEXES ROWS JUMP HEADER AND START FROM 1, COLUMNS START FROM 0
det_idx = det_idx.apply(pd.to_numeric) + [2, 0]

#ADD "JUST A DUMMY VALUE"
dummy_column =  np.full((det_idx.shape[0], 1), "JUST A DUMMY VALUE")
det_idx = pd.DataFrame(np.concatenate([det_idx, dummy_column], axis=1))

det_idx.to_csv(f'./DATASETS_REIN/{args.dataset}/detections.csv', header = False, index = False)
