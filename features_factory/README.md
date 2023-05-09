# github-PR-info-代码整理
# PR特征处理的代码

## requires
- mysqlclient
- pymongo

## 执行代码
- commit_info_50.py
- pr_info_50.py
- project_pr_50.py
- user_info_50.py


## 内容介绍-PR:50

### 爬取PR的commit信息

根据PR获取其commits信息，在~/pa_star_50_info/github_get_pull_commit 文件夹中。

```shell
python cmd.py
```

这个程序扫描`/data1/chenkexing/star_50/human_readable_pulls`文件夹下的PR文件目录信息，然后根据每一个PR文件中的commits_url链接获取commit信息，并保存到`/data1/chenkexing/star_50/pull_commits3`文件夹中。

### 爬取PR的commit details信息

由于上面获取的commit信息，不包括具体的代码修改信息，因此，需要根据上面获取的commit文件，获取更加具体的commit-detail信息。

具体程序在`~/pa_star_50_info/github_get_commit_detail`文件夹中。

首先需要修改cmd.py文件中的file_names内容。

```shell
python cmd.py
```

程序读取file_dir中对应commit文件，通过其中的api链接，将具体commit内容获取并保存在mongo数据库中。地址是mongo的`commit_details`库中以每个项目名命名的数据表。

### 一、PR特征-commit info 
处理PR中包含的commit相关特征信息
```shell
python commit_info_50.py
```
### 二、PR特征-pr info
处理PR本身的相关特征信息
```shell
python pr_info_50.py
```
### 三、PR特征-项目中包含的其他PR特征
统计项目中其他PR的关联特征信息
```shell
python project_pr_50.py
``` 

### 四、PR特征-PR作者特征
统计PR作者的特征
```
python pr_user_info_50.py
```

### 计算PR的merge信息
通过pulls和comment文件内容，判断每个项目pr的merge信息。
```shell
python pr_merge_info_50.py
```

### 合并特征信息
step 1: 在result的每个项目中，合并不同方面的特征。

step 2: 合并每个项目的特征，成为一个总的特征文件。
```shell
python gather_all_data.py
```

### checklist
#### project_pr
- [x] tongji_project_before_pr_issues :fix 只用pr id；添加issue个数特征
- [x] tongji_project_before_pr_prs
- [x] tongji_project_before_pr_comments_in_months
- [x] tongji_project_before_pr_comments
- [x] tongji_project_before_pr_commits_in_months
- [x] tongji_project_before_pr_commits

#### project_commit_detail
- [x] tongji_project_commit_total_change
- [x] commit message

#### user_info
- [x] cal_before_pr_user_project
- [x] cal_before_pr_user_commits_project
- [x] cal_before_pr_user_commits
- [x] cal_before_pr_user_prs
- [x] cal_before_pr_user_issues
- [x] cal_before_pr_user_followers

#### pr_info
- [x] call_pr_is_first => [ call_previous_pr ; call_previous_issues ] 
- [x] call_pr_has_good_description
- [x] get_pr_commit_count
- [ ] [Remove] get_pr_comment_count [no use in prediction]
- [ ] [Remove] call_pr_tags
