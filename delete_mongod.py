from pymongo import MongoClient

def connect_to_mongodb(collection_name):
    """Establish connection to MongoDB and return the specified collection."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['face_recognition_db']
        return db[collection_name]
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

def delete_face_document(collection, identifier_key, identifier_value):
    """Delete a document from the collection based on identifier_key and identifier_value."""
    try:
        result = collection.delete_one({identifier_key: identifier_value})
        return result.deleted_count
    except Exception as e:
        print(f"Error deleting document: {e}")
        return 0

def main():
    while True:
        collection_name = input("Enter the collection name to delete from (temp_faces or perm_faces): ").strip().lower()
        
        if collection_name not in ['temp_faces', 'perm_faces']:
            print("Invalid collection name. Please enter 'temp_faces' or 'perm_faces'.")
            continue
        
        identifier_key = 'face_id' if collection_name == 'temp_faces' else 'name'
        
        collection = connect_to_mongodb(collection_name)
        
        if collection is None:
            print("Failed to connect to MongoDB. Exiting.")
            return
        
        identifier_value = input(f"Enter the {identifier_key} to delete: ").strip()
        
        if not identifier_value:
            print(f"Error: {identifier_key} cannot be empty.")
            continue
        
        deleted_count = delete_face_document(collection, identifier_key, identifier_value)
        
        if deleted_count:
            print(f"Successfully deleted document with {identifier_key}: {identifier_value}")
        else:
            print(f"No document found with {identifier_key}: {identifier_value}")
        
        choice = input("Do you want to delete another document? (y/n): ").strip().lower()
        if choice != 'y':
            break

if __name__ == "__main__":
    main()
