
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
		--input_dir=XXXXXX.csv \
        --save_dir=./checkpoints/baseline-timeseries-10-fold-importance \
        --log_name=logs_XGBoost_importance_angular-angular.txt 


        # flutter/flutter
        # nodejs/node
        # tensorflow/models
        # kubernetes/kubernetes
        # angular/angular 