######################################
# csv、json file
######################################

import csv
import json
import os
import os.path


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def read_csv_data(file_name):
    csv.field_size_limit(500 * 1024 * 1024)
    with open(file_name, "r") as f:
        f_csv = csv.reader(f)
        flag = True
        data = []
        for row in f_csv:
            if flag:
                flag = False
                header = row
            else:
                data.append(row)
    return header, data


def read_csv_data_as_dict(file_name):
    res = []
    with open(file_name, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            res.append(row)
    return res


def write_csv_data_as_list_of_dict(file_name, dicts):
    # file_name = check_file_name(file_name)
    keys = dicts[0].keys()
    with open(file_name, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys, extrasaction="ignore")
        dict_writer.writeheader()
        dict_writer.writerows(dicts)


def check_file_name(file_name):
    append_suffix = 1
    while os.path.isfile(file_name):
        arr = file_name.split(".")
        suffix = arr[-1]
        name = ".".join(arr[0:-1])
        file_name = name + f"_{append_suffix}.{suffix}"
        append_suffix += 1
    return file_name


def write_csv_data(csv_file_name, header, data):
    # csv_file_name = check_file_name(csv_file_name)
    with open(csv_file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)


def write_json_data(data, json_file_name):
    # json_file_name = check_file_name(json_file_name)
    with open(json_file_name, "w") as f:
        json.dump(data, f, indent=2)
