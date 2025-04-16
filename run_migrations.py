import mysql
import mysql.connector
import os

DB_PARAMS = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "password",
    "database": "mrp_new"
}

SCHEMA_DIR = "schema"

def run_migrations():
    """Executes all SQL files in order from the schema folder."""
    try:
        conn = mysql.connector.connect(**DB_PARAMS)
        cursor = conn.cursor()

        sql_files = sorted(f for f in os.listdir(SCHEMA_DIR) if f.endswith(".sql"))

        for file in sql_files:
            file_path = os.path.join(SCHEMA_DIR, file)
            print(f"Running: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                sql_script = f.read()
                for statement in sql_script.split(";"):
                    if statement.strip():
                        cursor.execute(statement)

            print(f"{file} executed successfully!")

        conn.commit()
        cursor.close()
        conn.close()
        print("All migrations applied successfully!")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    run_migrations()
