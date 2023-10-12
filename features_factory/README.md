# PR features 

## requires
- mysqlclient
- pymongo

## releated file
- commit_info_50.py
- pr_info_50.py
- project_pr_50.py
- user_info_50.py

## content-PR:50

### a. PR manual feature commit info 

```shell
python commit_info_50.py
```
### b. PR manual feature pr info
pr self content info

```shell
python pr_info_50.py
```
### c. PR manual feature project info + other pr info
other pr info in the project

```shell
python project_pr_50.py
``` 

### d. PR manual feature PR creator info
pr author feature

```
python pr_user_info_50.py
```

### get PR merge info

```shell
python pr_merge_info_50.py
```

### gather all info

step 1: in result/ file, gather every project file.

step 2: gather every project feature to one.

```shell
python gather_all_data.py
```

### checklist

#### project_pr
- [x] tongji_project_before_pr_issues
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