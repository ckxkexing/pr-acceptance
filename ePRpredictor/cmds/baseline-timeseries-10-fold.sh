
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
        --add_text_data=True \
		--input_dir=/data1/kexingchen/epr_info_results_v7/all_projects_features_Z7.csv \
        --save_dir=./checkpoints/baseline-tmp \
        --log_name=logs_XGBoost.txt 
