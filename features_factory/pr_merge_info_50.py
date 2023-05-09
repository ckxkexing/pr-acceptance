'''
    
     Check if information is merged
    
'''

import re
import os
import json
from git import Repo
from use_ghtorrent.utils import read_json_data, write_csv_data


def is_sha1(maybe_sha):
    if len(maybe_sha) not in [7, 40, 12]:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True

def has_sha1_in_sentence(text):
    pattern = re.compile(r'\b[0-9a-f]{7,40}\b')   
    result = pattern.findall(text)
    result = ''.join(result)
    if len(result) > 0:
        return True
    else:
        return False

def has_pull_in_sentence(text):
    pattern = re.compile(r'\b\#(\d+)\b') # 匹配 '#2333' 模式。
    result = pattern.findall(text)
    result = ''.join(result)
    if len(result) > 0:
        return True

    pattern_url = re.compile(r'\bhttps://github.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\b')
    result = pattern_url.findall(text)
    result = ''.join(result)
    if len(result) > 0:
        return True
    return False

def check_merge_from_comment(text, login, author_association='NONE'):

    # check login is or not a bot
    patterns = [r'\Wbot\W$', r'bot$', r'\Wrobot$']
    for pattern in patterns:
        if re.findall(pattern, login):
            return 0

    text = text.lower()
    flag, ff = 0, 0
    text = text.replace('.', ' ')
    text = ' '.join(text.splitlines())
    text = text.split(' ')
    for i, word in enumerate(text):
        if word in ['but', 'cannot', 'donot', 'can\'t', 'don\'t','closing', 'before']:
            break
        if word in ['merged', 'landed', 'pushed', 'included', 'committed']:
            ff = 1
        if ff:
            next_sentence = " ".join(text[i:])
            if( has_sha1_in_sentence(next_sentence)):
                flag = 1
            if( has_pull_in_sentence(next_sentence)):
                flag = 1
            # if( author_association == 'CONTRIBUTOR'):
            #     flag = 1
            break
    # pattern = r'\Wlgtm\W'
    # text = ' '.join(text)
    # if len(text) < 10 and re.findall(pattern, text):
    #     flag = 1
    return flag

def get_repo_commit_mention_pull(msg):
    res = []
    msg = msg.splitlines()
    
    for text in msg:
        # 查找 start with gh- pr id
        pattern = re.compile(r'\bgh-(\d+)\b')   # type: " gh-233 "
        result = pattern.findall(text)
        res.extend(result)
        # 查找 start with # pr id
        pattern2 = re.compile(r'\b\#(\d+)\b')   # type " #2333 "
        result = pattern2.findall(text)
        # 查找 repo-commit 中pr url 
        pattern_url = re.compile(r'\bhttps://github.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\b')
        result = pattern_url.findall(text)
        for url in result:
            if 'pull' in url and url.split('/')[-1].isnumeric():
                res.append(url.split('/')[-1])
    return res

def cal_pr_merge_info(project_name):
    '''
        parms. example:
            project_name = moby_moby
    '''
    pulls_path = '/data1/chenkexing/star_50/human_readable_pulls/'
    project_pulls_file = os.path.join(pulls_path, f'human_pull_{project_name}.json')

    github_merge_map = {}
    with open(project_pulls_file, 'r') as f:
        pulls_data = json.load(f)
    github_cnt , total_pr = 0 , len(pulls_data)
    for pulls in pulls_data:
        if pulls['merged_at'] != None:
            id1 = pulls['url'].split('/')
            id1 = id1[id1.index('pulls') + 1]
            github_merge_map[id1] = 1
            github_cnt += 1

    print("PRs", len(pulls_data)) 
    print("github_merge_at:", github_cnt)

    comments_path = '/data1/chenkexing/star_50/pull_comment'
    project_comments_file = os.path.join(comments_path, f'{project_name}_pr_comments.json')

    comment_cnt = 0
    with open(project_comments_file, 'r') as f:
        comments_data = json.load(f)
    comment_merge_map = {}

    # judge pr merge by pulls comments
    for data in comments_data:
        flag = 0
        for d in data['comments'][-3:]:  #[-3:]
            if(check_merge_from_comment(d['body'], d['user']['login'], d['author_association'])):
                flag = 1
        comment_cnt += flag
        if flag:
            id2 = data['pull_url'].split('/')
            id2 = id2[id2.index('issues') + 1]
            comment_merge_map[id2] = 1
    print("comment_merge_at:", comment_cnt)


    #  By repo 's commit
    commit_path = '/data1/chenkexing/star_50/repo_commits'
    repo_commits_fille = os.path.join(commit_path, f'{project_name}_repo_commits.json')
    commit_mention_cnt = 0
    commit_included_cnt = 0

    with open(repo_commits_fille, 'r') as f:
        repo_commits_data = json.load(f)

    #   Merge if be mentioned by commits
    commit_mention_map = {}
    for data in repo_commits_data:
        msg = data['commit']['message']
        ids = get_repo_commit_mention_pull(msg)
        for id in ids:
            commit_mention_map[id] = 1
    
    #   Merge if be included in repo commits
    commit_included_map = {}
    included_commit = {}
    clone_dir = '/data1/chenkexing/star_50/clone_star_50_projects_only_commits'
    repo_clone_dir = os.path.join(clone_dir, project_name.replace('_', '/')) + '/.git'
    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits('--all', max_count=None)
    for commit in all_commits:
        included_commit[commit.hexsha] = 1
        msg = commit.message
        ids = get_repo_commit_mention_pull(msg)
        for id in ids:
            commit_mention_map[id] = 1
    
    pr_commits_dir = f'/data1/chenkexing/star_50/pull_commits3/{project_name}_pr_commits.json'
    with open(pr_commits_dir, 'r') as f:
        pr_commits_data = json.load(f)
    for pr in pr_commits_data:
        id1 = pr['pull_url'].split('/')
        id1 = id1[id1.index('pulls') + 1]
        flag = 0
        for commit in pr['commits']:
            if commit['sha'] in included_commit:
                flag += 1
        if flag and flag == len(pr['commits']):
            commit_included_map[id1] = 1

    header = ['pr_url', 'merge', 'github_merge', 'comment_merge', 'commit_merge']
    res = []
    total_merge = 0
    
    for pulls in pulls_data:
        cur = []
        cur.append(pulls['url'])
        ix = pulls['url'].split('/')
        ix = ix[ix.index('pulls') + 1]
        a = 0
        if ix in github_merge_map:
            a = 1
            # 有的commit会在后来被合并到同一个commit中，即原来的commit会删除。
            # 因此含有commit的数目可能会小于github merge。
            # if ix not in commit_included_map:
            #     print(pulls['url'])
        b = 0
        if ix in comment_merge_map:
            b = 1
        c = 0
        if ix in commit_mention_map or ix in commit_included_map:
            c = 1
            if ix in commit_mention_map:
                commit_mention_cnt += 1
            elif ix in commit_included_map:
                commit_included_cnt += 1

        merge = 0
        if a + b + c:
            total_merge += 1
            merge = 1
        cur.extend([merge, a, b, c])
        res.append(cur)
    print("commit_mention_cnt: ", commit_mention_cnt)
    print("commit_include_cnt: ", commit_included_cnt)
    print("total_merge:", total_merge)
    output_path = f'./results/result/{project_name}'
    output_file = os.path.join(output_path, f'{project_name}_merge_info.csv')
    write_csv_data(output_file, header, res)
    # logger.info(f"{project_name}\t{github_cnt}\t{comment_cnt}\t{commit_mention_cnt}")
    return github_cnt , comment_cnt, commit_mention_cnt, commit_included_cnt, total_pr, total_merge


if __name__ == '__main__':

    data = read_json_data('star_proj_get_2021_9_7_pre50.json')
    info_header = ['project_name', 'github_cnt','comment_cnt', 'commit_mention_cnt', 'commit_included_cnt', 'total_pr', 'total_merge']
    res = []
    for proj in data:
        # if proj['name'] != 'angular/angular':
        #     continue
        print(f"正在处理{proj['name']}")
        project_name = proj['name'].replace('/', '_')
        github_cnt , comment_cnt, commit_mention_cnt, commit_included_cnt, total_pr, total_merge = cal_pr_merge_info(project_name)
        res.append([project_name, github_cnt , comment_cnt, commit_mention_cnt, commit_included_cnt, total_pr, total_merge])

    # write_csv_data('50_proj_merge_info.csv', info_header, res)
    
    # 下面是测试代码
    # print(check_merge_from_comment('xxx', "fluttergithubbot"))
    # print(check_merge_from_comment("@annthurium are you sure you committed package.json?  Isn't showing up for me in the diff.", "MEMBER"))
    
    # print(check_merge_from_comment('''\nThanks for your pull request. It looks like this may be your first contribution to a Google open source project (if not, look below for help). Before we can look at your pull request, you'll need to sign a Contributor License Agreement (CLA).\n\n:memo: **Please visit <https://cla.developers.google.com/> to sign.**\n\nOnce you've signed (or fixed any issues), please reply here with `@googlebot I signed it!` and we'll verify it.\n\n----\n\n#### What to do if you already signed the CLA\n\n##### Individual signers\n\n*   It's possible we don't have your GitHub username or you're using a different email address on your commit. Check [your existing CLA data](https://cla.developers.google.com/clas) and verify that your [email is set on your git commits](https://help.github.com/articles/setting-your-email-in-git/).\n\n##### Corporate signers\n\n*   Your company has a Point of Contact who decides which employees are authorized to participate. Ask your POC to be added to the group of authorized contributors. If you don't know who your Point of Contact is, direct the Google project maintainer to [go/cla#troubleshoot](http://go/cla#troubleshoot) ([Public version](https://opensource.google/docs/cla/#troubleshoot)).\n*   The email used to register you as an authorized contributor must be the email used for the Git commit. Check [your existing CLA data](https://cla.developers.google.com/clas) and verify that your [email is set on your git commits](https://help.github.com/articles/setting-your-email-in-git/).\n*   The email used to register you as an authorized contributor must also be [attached to your GitHub account](https://github.com/settings/emails).\n\t\t\n\n\u2139\ufe0f **Googlers: [Go here](https://goto.google.com/prinfo/https%3A%2F%2Fgithub.com%2Fangular%2Fangular%2Fpull%2F43385) for more info**.\n\n<!-- need_sender_cla -->''', "google-cla[bot]"))

    # print(check_merge_from_comment("Superseded by #18300", "CONTRIBUTOR"))
    
    
    c = 5 / 0



    # print(check_merge_from_comment("Landed in 7e7062cdca87e5ff54945fc1786cba25d0996995.Thanks for the contribution! 🎉"))
    
    # a = check_merge_from_comment('This PR was merged into the repository by commit 9c03b6371a96c63316623b8820960b4b5d494bb4.')
    # print('wola~', a)

    msg = 'lgtm'
    print('lgtm', check_merge_from_comment(msg))
    '''
    print(is_sha1('1b47866a1d94224678ce05e7c7355ff17ad9cce6'))
    print(is_sha1('338ab0dfa0c9'))
    print(int('338ab0dfa0c9', 16))

    msg = "Initialize NativeDetector at build time\n\nCloses gh-28244 gh-3333"
    msg = "Change link from 5.3.x to main\n\nSee gh-28228"
    print(get_repo_commit_mention_pull(msg))
    '''
    msg = '''
        docs(bazel): fix outdated redirect URL for `/guide/bazel` (#43376)

        The file we are redirecting `/guide/bazel` to was moved from
        `bazel/src/schematics/README.md` to `bazel/docs/BAZEL_SCHEMATICS.md` in
        commit 71b8c9ab29014f7e710e03ebda185c0a7c0c2620.

        Update the Firebase configuration to use the new path in the redirect
        URL.

        PR Close #43376
    '''
    print(has_sha1_in_sentence(msg))
    msg = '''
        build: initialize variable before use\n\nfound with make --warn-undefined-variables\n\nPR-URL: https://github.com/iojs/io.js/pull/320\nReviewed-By: Rod Vagg <rod@vagg.org>
    '''
    print(get_repo_commit_mention_pull(msg))

    msg = '''
    Drag and drop to install apk files from computer\n\n<https://github.com/Genymobile/scrcpy/pull/133>
    '''
    print(get_repo_commit_mention_pull(msg))
