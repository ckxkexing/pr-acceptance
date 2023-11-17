from utils.mysql_link import run_sql_mysqlclient

mq = "select * from users a join commits b on a.id=b.author_id where a.login='bprashanth';"
res = run_sql_mysqlclient(mq)

new_res = []
for i, s in enumerate(res):
    if isinstance(s["created_at"], str) or s["created_at"] is None:
        continue
    new_res.append(s)


def key(a):
    return a["b.created_at"]


print(new_res[0])
print(new_res[1])
new_res.sort(key=key)
print("\n" * 10)
# print(new_res)
print(new_res[0])
print(new_res[1])
