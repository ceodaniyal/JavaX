#app/db/mongodb.py
import os
import logging
from datetime import datetime, timezone
from pymongo import MongoClient
import gridfs
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL")
DB_NAME = "Javax"
COLLECTION_NAME = "chats"

client = None
db = None
chats_collection = None
fs = None

try:
    if MONGODB_URL:
        client = MongoClient(MONGODB_URL)
        db = client[DB_NAME]
        chats_collection = db[COLLECTION_NAME]
        fs = gridfs.GridFS(db)
        logger.info("MongoDB connection established successfully to %s.%s", DB_NAME, COLLECTION_NAME)
    else:
        logger.warning("MONGODB_URL not found in environment variables.")
except Exception as e:
    logger.error("Failed to connect to MongoDB: %s", str(e))

def save_file(contents: bytes, filename: str) -> str:
    if fs is None:
        return None
    try:
        file_id = fs.put(contents, filename=filename)
        return str(file_id)
    except Exception as e:
        logger.error("Error saving file to GridFS: %s", str(e))
        return None

def get_file(file_id: str) -> bytes:
    if fs is None:
        return None
    try:
        f = fs.get(ObjectId(file_id))
        return f.read()
    except Exception as e:
        logger.error("Error getting file from GridFS: %s", str(e))
        return None

def get_filename(file_id: str) -> str:
    if fs is None:
        return None
    try:
        f = fs.get(ObjectId(file_id))
        return f.filename
    except Exception as e:
        logger.error("Error getting filename from GridFS: %s", str(e))
        return None

def save_analysis(task_id: str, query: str, filename: str, dataset_id: str = None, file_id: str = None):
    """Initialises a history record with task_id and query."""
    if chats_collection is None:
        return
    
    try:
        # Check if dataset session already exists
        existing = chats_collection.find_one({"dataset_id": dataset_id}) if dataset_id else None
        
        if existing:
            chats_collection.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "current_task_id": task_id,
                    "current_query": query,
                    "status": "running",
                    "updated_at": datetime.now(timezone.utc)
                }}
            )
            logger.info("Updated existing session for dataset %s with new task %s", dataset_id, task_id)
        else:
            record = {
                "task_id": task_id, # Keep for backwards compatibility
                "current_task_id": task_id,
                "query": query, # Initial query
                "current_query": query,
                "filename": filename,
                "dataset_id": dataset_id,
                "file_id": file_id,
                "status": "running",
                "created_at": datetime.now(timezone.utc),
                "data": None,
                "interactions": [],
                "chats": []
            }
            chats_collection.insert_one(record)
            logger.info("Saved initial analysis record for task %s", task_id)
    except Exception as e:
        logger.error("Error saving initial analysis: %s", str(e))

def update_analysis_result(task_id: str, status: str, result_data: dict = None, error: str = None):
    """Updates the analysis record with results or error."""
    if chats_collection is None:
        return
    
    try:
        doc = chats_collection.find_one({"$or": [{"current_task_id": task_id}, {"task_id": task_id}]})
        if not doc:
            logger.error("Could not find document for task %s", task_id)
            return

        update_doc = {
            "$set": {
                "status": status,
                "updated_at": datetime.now(timezone.utc)
            }
        }
        
        if status == "completed" and result_data:
            update_doc["$set"]["data"] = result_data # Keep latest data for backwards compat
            
            # Append to interactions
            interaction = {
                "query": doc.get("current_query", doc.get("query", "")),
                "data": result_data,
                "timestamp": datetime.now(timezone.utc)
            }
            update_doc["$push"] = {"interactions": interaction}
            
        if error:
            update_doc["$set"]["error"] = error
            
        chats_collection.update_one({"_id": doc["_id"]}, update_doc)
        logger.info("Updated analysis result for task %s with status %s", task_id, status)
    except Exception as e:
        logger.error("Error updating analysis result: %s", str(e))

def append_chat(dataset_id: str, query: str, answer: str, table: list = None):
    """Appends a chat interaction to the analysis record."""
    if chats_collection is None:
        return
    
    try:
        chat_entry = {
            "query": query,
            "answer": answer,
            "table": table,
            "timestamp": datetime.now(timezone.utc)
        }
        chats_collection.update_one(
            {"dataset_id": dataset_id},
            {"$push": {"chats": chat_entry}}
        )
    except Exception as e:
        logger.error("Error appending chat: %s", str(e))

def get_history(limit: int = 50):
    """Fetches the latest completed/error analyses."""
    if chats_collection is None:
        return []
    
    try:
        # Fetching latest first. We exclude data and interactions to reduce payload size.
        cursor = chats_collection.find({}, {"_id": 0, "data": 0, "interactions": 0}).sort("created_at", -1).limit(limit)
        results = []
        for doc in cursor:
            # Ensure task_id is present for the sidebar
            doc["task_id"] = doc.get("current_task_id", doc.get("task_id"))
            
            if "created_at" in doc and isinstance(doc["created_at"], datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            if "updated_at" in doc and isinstance(doc["updated_at"], datetime):
                doc["updated_at"] = doc["updated_at"].isoformat()
            results.append(doc)
        return results
    except Exception as e:
        logger.error("Error fetching history: %s", str(e))
        return []

def _convert_datetimes(obj):
    """Recursively convert all datetime objects in a dict/list to ISO strings."""
    if isinstance(obj, dict):
        return {k: _convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_datetimes(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

def get_analysis_by_id(task_id: str):
    """Fetches a specific analysis by its task_id."""
    if chats_collection is None:
        return None
    
    try:
        doc = chats_collection.find_one({"$or": [{"current_task_id": task_id}, {"task_id": task_id}]}, {"_id": 0})
        if doc:
            # Backwards compatibility: inject interactions if missing
            if "interactions" not in doc:
                if doc.get("data"):
                    doc["interactions"] = [{
                        "query": doc.get("query", ""),
                        "data": doc.get("data"),
                        "timestamp": doc.get("created_at")
                    }]
                else:
                    doc["interactions"] = []
            doc = _convert_datetimes(doc)
        return doc
    except Exception as e:
        logger.error("Error fetching analysis by id %s: %s", task_id, str(e))
        return None

def get_analysis_by_dataset_id(dataset_id: str):
    if chats_collection is None:
        return None
    
    try:
        return chats_collection.find_one({"dataset_id": dataset_id}, {"_id": 0})
    except Exception as e:
        logger.error("Error fetching analysis by dataset id %s: %s", dataset_id, str(e))
        return None

def delete_analysis(task_id: str):
    """Deletes an analysis and its associated file from GridFS."""
    if chats_collection is None:
        return False
    
    try:
        # First, find it to get file_id
        doc = chats_collection.find_one({"task_id": task_id}, {"file_id": 1})
        if doc and doc.get("file_id") and fs is not None:
            try:
                fs.delete(ObjectId(doc["file_id"]))
            except Exception as e:
                logger.error("Error deleting file from GridFS: %s", str(e))
                
        result = chats_collection.delete_one({"task_id": task_id})
        return result.deleted_count > 0
    except Exception as e:
        logger.error("Error deleting analysis %s: %s", task_id, str(e))
        return False
