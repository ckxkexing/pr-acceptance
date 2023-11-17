import yaml

f = open("configs.yaml", "r")
config = yaml.load(f.read(), Loader=yaml.SafeLoader)
