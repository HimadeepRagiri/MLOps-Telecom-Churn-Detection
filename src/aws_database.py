import boto3
import os
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv
from decimal import Decimal

load_dotenv()
AWS_REGION = os.getenv("AWS_REGION")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE")

def get_dynamodb_resource():
    return boto3.resource("dynamodb", region_name=AWS_REGION)

def store_prediction(features: dict, prediction: int, proba: float):
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(DYNAMODB_TABLE)
    # Use CustomerID from features, or generate one if not present
    customer_id = features.get("CustomerID", f"customer_{datetime.utcnow().timestamp()}")
    prediction_date = datetime.utcnow().isoformat()
    item = features.copy()
    item["CustomerID"] = customer_id
    item["PredictionDate"] = prediction_date
    item["Churn"] = "Yes" if prediction == 1 else "No"
    item["probability"] = Decimal(str(proba))

    # Convert all float values in item to Decimal
    for k, v in item.items():
        if isinstance(v, float):
            item[k] = Decimal(str(v))

    try:
        table.put_item(Item=item)
        logger.info("Stored prediction in DynamoDB.")
    except Exception as e:
        logger.error(f"Failed to store in DynamoDB: {e}")

def get_all_production_data():
    """
    Get all production data points, sorted by PredictionDate ascending.
    """
    dynamodb = get_dynamodb_resource()
    table = dynamodb.Table(DYNAMODB_TABLE)
    response = table.scan()
    items = response.get("Items", [])
    # Sort by PredictionDate ascending (oldest first)
    items = sorted(items, key=lambda x: x.get("PredictionDate", ""))
    return items