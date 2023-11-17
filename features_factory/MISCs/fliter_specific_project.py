import pandas as pd

train_file = "/home/chenkx/github-pr-info-50/results/all_projects_features_Z2.csv"

pr_data = pd.read_csv(train_file, low_memory=False)

pr_data = pr_data[
    pr_data["pr_url"].str.contains("https://api.github.com/repos/pytorch/pytorch")
]

pr_data.to_csv(
    "/home/chenkx/github-pr-info-50/results/all_projects_features_Z2_pytorch_pytorch.csv",
    index=False,
    sep=",",
    encoding="utf-8",
)
