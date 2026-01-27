#!/bin/bash

# Configuration
DB_NAME="agrarian_db"
DB_ADMIN_USERNAME="admin"
DB_ADMIN_PASSWORD="admin_pass"
APP_SERVICE_USER="app_manager"
APP_SERVICE_PASS="app_manager_pass"
CONTAINER_NAME="postgres_db"
DATA_DIR="$(pwd)/postgres_data"

DB_USERNAME=testuser@testmail.com
DB_PASSWORD=testpassword

HASHED_PASS='$2b$12$FG4O4lGlmO/O9CUedLzyb./ulIMOwHVqJkOfeNdQEWGiyGFOzD0y6'

# 1. Clean up
echo "Cleaning up..."
docker rm -f $CONTAINER_NAME || true
mkdir -p "$DATA_DIR"

echo "Starting PostgreSQL (v18+ compatible)..."

# 2. Launch with updated mount point
docker run -d \
  --name $CONTAINER_NAME \
  -e POSTGRES_DB=$DB_NAME \
  -e POSTGRES_USER=$DB_ADMIN_USERNAME \
  -e POSTGRES_PASSWORD=$DB_ADMIN_PASSWORD \
  -p 5432:5432 \
  -v "$DATA_DIR:/var/lib/postgresql" \
  postgres:latest

# 3. Improved Health Check
echo -n "Waiting for Postgres engine to be ready..."
# pg_isready uses -U for user and -d for database
until docker exec $CONTAINER_NAME pg_isready -U "$DB_ADMIN_USERNAME" -d "$DB_NAME" > /dev/null 2>&1; do
    echo -n "."
    sleep 1
done
echo -e "\nEngine is up. Finalizing '$DB_NAME'..."

# 4. Configure everything in one block
echo "Initializing Database: Roles, Schema, and Permissions..."
docker exec -i $CONTAINER_NAME psql -U "$DB_ADMIN_USERNAME" -d "$DB_NAME" <<EOF
-- 1. Create Role only if it doesn't exist
DO \$$ 
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$APP_SERVICE_USER') THEN
        CREATE ROLE $APP_SERVICE_USER WITH LOGIN PASSWORD '$APP_SERVICE_PASS';
    END IF;
END \$$;

-- 2. Create Tables (Ordered by dependency)
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(128) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    flight_id INTEGER NOT NULL REFERENCES flights(flight_id) ON DELETE CASCADE,
    alert_msg TEXT NOT NULL,
    frame_id INTEGER,
    timestamp DOUBLE PRECISION,
    datetime TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    image_data BYTEA,
    image_width INTEGER,
    image_height INTEGER
);

-- 3. Grant Permissions
GRANT USAGE ON SCHEMA public TO $APP_SERVICE_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO $APP_SERVICE_USER;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO $APP_SERVICE_USER;

-- 4. Future-proofing: Grant permissions on future tables automatically
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO $APP_SERVICE_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO $APP_SERVICE_USER;

-- 5. Inject Test User
INSERT INTO users (email, password) 
VALUES ('$DB_USERNAME', '$HASHED_PASS') 
ON CONFLICT (email) DO UPDATE 
SET password = EXCLUDED.password, 
    created_at = CURRENT_TIMESTAMP;
EOF

echo "------------------------------------------------"
echo "SUCCESS: Database configured."
echo "App Service User: $APP_SERVICE_USER"
echo "Test User: $DB_USERNAME"
echo "------------------------------------------------"



# List all tables
# docker exec -it postgres_db psql -U admin -d agrarian_db -c "\dt"

# View the structure (columns, types) of the 'users' table
# docker exec -it postgres_db psql -U admin -d agrarian_db -c "\d users"

# users table content
# docker exec -it postgres_db psql -U admin -d agrarian_db -c "SELECT * FROM users;"

# permissions
# docker exec -it postgres_db psql -U admin -d agrarian_db -c "\du"