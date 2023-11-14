
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
        --project_name=angular/angular \
        --time_series=True \
        --log_importance_photo=logs_XGBoost_importance_angular-angular.pdf \
        --add_text_data=False \
		--input_dir=/home/chenkx/github-pr-info-50/results/all_projects_features_Z2_in_lifetime_sorted.csv \
        --save_dir=./checkpoints/baseline-timeseries-10-fold-importance \
        --log_name=logs_XGBoost_importance_angular-angular.txt 


        # flutter/flutter
        # nodejs/node
        # tensorflow/models
        # kubernetes/kubernetes
        # angular/angular 