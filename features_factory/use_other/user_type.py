'''

    check a user is bot or not by user's login

'''
import os
import re
import json
from .utils import Timer
from .utils import write_csv_data
from tqdm import tqdm

def bot_detect_by_login(file_name, project_name, csv_file_path):
    '''
        file_name = {proj}_pull.json
    '''

    job = 'detect_user_type'
    timer = Timer()
    logins = []
    pulls = []
    with open(file_name, 'r') as f:
        json_data = json.load(f)
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            login = data['user']['login']
            tmp['login'] = login
            pulls.append(tmp)

            if 'bot' in login and login not in logins:
                logins.append(login)

            
    # print("[only bot] ", " ; ".join(logins))

    new_logins = []
    for login in logins:
        # 结尾是[bot],-bot,-robot的名字，认定为bot
        patterns = [r'\Wbot\W$', r'\Wbot$', r'\Wrobot$']
        for pattern in patterns:
            if re.findall(pattern, login):
                new_logins.append(login)
                break

    # print("[only \Wbot\W] ", " ; ".join(new_logins))
    res_header = ['pr_url', 'bot_user']
    res = []
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        if pr['login'] not in new_logins:
            cur.append("0")
        else:
            cur.append("1")
        res.append(cur)
    
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}.csv')
    write_csv_data(csv_file, res_header, res)
    
