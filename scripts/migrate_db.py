import os
import psycopg2
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def migrate():
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "invoice_db"),
        user=os.getenv("POSTGRES_USER", "invoice_user"),
        password=os.getenv("POSTGRES_PASSWORD", "invoice_pass"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        hashed_password VARCHAR(255) NOT NULL
    );
    """)
    
    cursor.execute("""
    ALTER TABLE invoices ADD COLUMN IF NOT EXISTS confirmed_by_user_id INTEGER REFERENCES users(id) DEFAULT NULL;
    """)
    cursor.execute("""
    ALTER TABLE invoices ADD COLUMN IF NOT EXISTS confirmed_at TIMESTAMPTZ DEFAULT NULL;
    """)
    
    cursor.execute("SELECT id FROM users WHERE username = 'admin';")
    if not cursor.fetchone():
        hashed_pw = pwd_context.hash("admin")
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (%s, %s);", ("admin", hashed_pw))
        print("Admin user created")
    
    print("Migration successful")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    migrate()
