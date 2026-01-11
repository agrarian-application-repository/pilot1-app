from sqlalchemy import create_engine, inspect, text

engine = create_engine("sqlite:///alerts.db")
inspector = inspect(engine)

print(f"Tables: {inspector.get_table_names()}")

with engine.connect() as conn:
    # Wrap the string in text()
    result = conn.execute(text("SELECT * FROM users")) 
    
    for row in result:
        # row is a Row object, you can access by index or column name
        print(row)

    result = conn.execute(text("SELECT * FROM flights")) 
    
    for row in result:
        # row is a Row object, you can access by index or column name
        print(row)
        
    result = conn.execute(text("SELECT * FROM alerts")) 
    
    for row in result:
        # row is a Row object, you can access by index or column name
        print(row)

    
