####################
# Get Info Related PR Creator
####################
import datetime
import json


def read_pulls(json_file):
    pulls = []
    with open(json_file, "r") as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            data = json.loads(lin)
            pulls.append(
                {
                    "pr": data["url"],
                    "pr_id": data["id"],
                    "login": data["user"]["login"],
                    "date": datetime.datetime.strptime(
                        data["created_at"], "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "closed_at": datetime.datetime.strptime(
                        data["closed_at"], "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
    return pulls
