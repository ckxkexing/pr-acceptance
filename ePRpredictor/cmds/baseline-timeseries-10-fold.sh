
#
#  classifier :
#
#
#       [LogisticRegression]
#       [DecisionTreeClassifier]
#       [RandomForestClassifier]
#       [XGBoost]
#
python baseline_k_fold.py \
        --classifier=XGBoost \
        --nominize_in_project=False \
        --time_series=True \
        --add_text_data=False \
		--input_dir=XXXXXX.csv \
        --save_dir=./checkpoints/baseline-timeseries-10-fold \
        --log_name=logs_XGBoost.txt 