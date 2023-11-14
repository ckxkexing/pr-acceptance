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
        --time_series=False \
        --add_text_data=False \
        --input_dir=/home/chenkx/github-pr-info-50/results/all_projects_features_Z2_with_code_diff.csv \
        --save_dir=./checkpoints/baseline-tmp \
        --log_name=logs_RandomForestClassifier.txt