from kafka import KafkaConsumer
import json
from pymongo import MongoClient

def save_data():
    
    client = MongoClient('localhost', 27017)
    db = client['data_analytics_db']  
    collection_name = 'sarcasm'

    if collection_name in db.list_collection_names():
        db.drop_collection(collection_name)



    collection = db[collection_name]  

    
    consumer = KafkaConsumer(
        'my_topic',
        bootstrap_servers='localhost:9092',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    for message in consumer:
        document = message.value
        collection.insert_one(document)  


if __name__ == "__main__":
    save_data()