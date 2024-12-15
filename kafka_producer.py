import pandas as pd
from kafka import KafkaProducer
import json
import time
import logging

def send_data():

    csv_file_path = 'datasets/sarcasm.csv'
    df = pd.read_csv(csv_file_path)

    time.sleep(40)

    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda x: json.dumps(x).encode('utf-8')  
    )


    for index, row in df.iterrows():
        data = row.to_dict()  
        producer.send('my_topic', value=data) 

    
    producer.flush()  
    producer.close()  

if __name__ == "__main__":
    send_data()
