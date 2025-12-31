from src.database.models import SessionLocal, UserInteraction

db = SessionLocal()
try:
    deleted = db.query(UserInteraction).delete()
    db.commit()
    print(f"Deleted {deleted} interactions.")
finally:
    db.close()
