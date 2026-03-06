import os

db_file = "chatbot.db"  # replace with your DB filename

if os.path.exists(db_file):
    os.remove(db_file)
    print("Database deleted!")
else:
    print("Database file not found.")
