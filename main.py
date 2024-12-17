from kafka_producer import send_data
from mongodb_consumer import save_data
from spark_consumer import process_data

import threading

if __name__ == '__main__':
    spark_thread = threading.Thread(target=process_data)
    spark_thread.start()

    mongodb_thread = threading.Thread(target=save_data)
    mongodb_thread.start()

    send_data()

    spark_thread.join()
    mongodb_thread.join()

