import pymongo

# Connect to MongoDB.
host = "127.0.0.1"
port = 27017
client = pymongo.MongoClient(host=host, port=port)
