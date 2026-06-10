from kafka import KafkaProducer # type: ignore
import json, time, random

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

while True:
    transaction = {
        "Time": random.uniform(0, 172792),
        "Amount": random.uniform(1, 2000),
        **{f"V{i}": random.uniform(-5, 5) for i in range(1, 29)}
    }
    producer.send("transactions", transaction)
    print("Sent:", transaction)
    time.sleep(1)
