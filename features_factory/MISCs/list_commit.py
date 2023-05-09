'''
    test for list a cloned project's all commits

'''

import git
from git import Repo
import time

# path = '/data1/chenkexing/github_projects/projects/knex/knex/.git'
path = '/data1/chenkexing/star_50/clone_star_50_projects_only_commits/django/django/.git'
repo = Repo(path)

commits = repo.iter_commits('--all', max_count=None)
cnt = 0
mp = {}


for commit in commits:
    #   print("Committed by %s on %s with sha %s" % (commit.committer.name, time.strftime("%a, %d %b %Y %H:%M", time.localtime(commit.committed_date)), commit.hexsha)) 
    cnt += 1
    # print("#")
    # print(commit.committed_date)
    # print(commit.message)
    # print(commit.author.name)
    # print(commit.author.email)
    
    mp[commit.hexsha] = 1

    # if cnt > 5:
    #     c = 5 / 0
if 'eaa248b19d6f8dc8b750a170c4b088c5d357c18f' in mp:
    print("OK")
else:
    print("No")

print(cnt)
print(len(mp))