#
#  classifier :
#
#       [LogisticRegression]
#       [DecisionTreeClassifier]
#       [RandomForestClassifier]
#       [XGBoost]
#
python baseline_k_fold.py \
        --classifier=RandomForestClassifier \
        --nominize_in_project=False \
        --time_series=False \
        --add_text_data=False \
        --input_dir=XXXXXX.csv \
        --save_dir=./checkpoints/baseline-random-10-fold \
        --log_name=logs_RandomForestClassifier.txt