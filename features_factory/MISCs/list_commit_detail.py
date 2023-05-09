'''
    test for list details of commits.
    Especially commits' filenames
'''

import git
from git import Repo
import time
from pydriller import Repository

# path = '/data1/chenkexing/github_projects/projects/knex/knex/.git'
# path = '/data1/chenkexing/star_50/clone_star_50_projects_only_commits/django/django/.git'
path = '/data1/chenkexing/star_50/clone_star_50_projects/django/django/.git'
path = '/data1/chenkexing/star_50/clone_star_50_projects/scrapy/scrapy/.git'
path = '/data1/chenkexing/star_50/clone_star_50_projects/vercel/next.js/.git'
repo = Repo(path)

commits = repo.iter_commits('--all', max_count=None)

cnt = 0
mp = {}


driller_cnt = 0
for commit in Repository(path, include_refs = True, include_remotes=True).traverse_commits():

    # print(commit.modified_files)
    # print("date = ", commit.author_date)
    # print(commit.author_timezone )
    # print("files = ", commit.files)

    for m in commit.modified_files:
        # print(
        #     "Author {}".format(commit.author.name),
        #     " modified {}".format(m.filename),
        #     "\n new path {} \n".format(m.new_path),
        #     "\n old path {} \n".format(m.old_path),
        #     " with a change type of {}".format(m.change_type.name),
        #     " and the complexity is {}".format(m.complexity)
        # )
        if 'RE' in m.change_type.name:
            print(m.change_type.name)
        # print("nloc = ", m.nloc)    # 如果nloc ！= None 则认定为 source code 文件。
        # print(len(m.methods))
    if driller_cnt > 1000:
        c = 5 / 0
    driller_cnt += 1

c = 10 / 0

gitpython_cnt = 0
for commit in commits:
    gitpython_cnt += 1
    #   print("Committed by %s on %s with sha %s" % (commit.committer.name, time.strftime("%a, %d %b %Y %H:%M", time.localtime(commit.committed_date)), commit.hexsha)) 
    cnt += 1
    # print("#")
    # print(commit.committed_date)
    # print(commit.message)
    # print(commit.author.name)
    # print(commit.author.email)
    
    # mp[commit.hexsha] = 1
    # for data in commit.stats.files:
    #     print(data)

print(cnt)
c = 17 / 0
print(driller_cnt)
print(gitpython_cnt)
if 'eaa248b19d6f8dc8b750a170c4b088c5d357c18f' in mp:
    print("OK")
else:
    print("No")

print(cnt)
print(len(mp))