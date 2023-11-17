"""

    check a user is bot or not by user's login

"""
import os
import re

from tqdm import tqdm

from utils.common import read_pulls
from utils.files import write_csv_data


def bot_detect_by_login(file_name, project_name, csv_file_path):
    """
    "bot_user"
    """
    job = "detect_user_type"
    logins = []
    pulls = []
    pulls = read_pulls(file_name)
    for pr in pulls:
        login = pr["login"]
        if "bot" in login and login not in logins:
            logins.append(login)

    new_logins = []
    for login in logins:
        # Ends with the name of [bot], -bot, -robot, identified as bot
        patterns = [r"\Wbot\W$", r"\Wbot$", r"\Wrobot$"]
        for pattern in patterns:
            if re.findall(pattern, login):
                new_logins.append(login)
                break

    res_header = ["pr_url", "bot_user"]
    res = []
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        if pr["login"] not in new_logins:
            cur.append("0")
        else:
            cur.append("1")
        res.append(cur)

    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}.csv")
    write_csv_data(csv_file, res_header, res)
