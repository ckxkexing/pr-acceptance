import json

import pymongo

host = "127.0.0.1"
port = 27017
client = pymongo.MongoClient(host=host, port=port)
db = client["clone_projects"]
col = db["items"]

with open("../star_proj_get_2021_9_7_pre50.json", "r") as f:
    json_data = json.load(f)

for data in json_data:
    project_name = data["name"]
    cur = col.find_one({"name": project_name})
    if not cur:
        print(project_name)
