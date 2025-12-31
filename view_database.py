"""
View database contents
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.database.models import SessionLocal, Paper, Article, UserInteraction
from datetime import datetime

def view_database():
    """View all database contents"""
    db = SessionLocal()
    
    try:
        print("=" * 80)
        print("DATABASE CONTENTS")
        print("=" * 80)
        
        # User Interactions
        interactions = db.query(UserInteraction).order_by(UserInteraction.timestamp.desc()).all()
        print(f"\nUSER INTERACTIONS ({len(interactions)} total)")
        print("-" * 80)
        if interactions:
            for i, interaction in enumerate(interactions[:50], 1):  # Show first 50
                print(f"{i}. {interaction.interaction_type.upper():10} | "
                      f"Type: {interaction.item_type:7} | "
                      f"ID: {interaction.item_id:5} | "
                      f"Time: {interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            if len(interactions) > 50:
                print(f"... and {len(interactions) - 50} more interactions")
        else:
            print("No interactions yet")
        
        # Papers
        papers = db.query(Paper).order_by(Paper.collected_date.desc()).all()
        print(f"\nPAPERS ({len(papers)} total)")
        print("-" * 80)
        recommended_papers = [p for p in papers if p.recommended]
        print(f"Recommended: {len(recommended_papers)}")
        print(f"Not recommended: {len(papers) - len(recommended_papers)}")
        if papers:
            print("\nRecent papers:")
            for paper in papers[:10]:  # Show first 10
                status = "[RECOMMENDED]" if paper.recommended else "[NOT RECOMMENDED]"
                print(f"  - {paper.title[:60]:60} | {status}")
        
        # Articles
        articles = db.query(Article).order_by(Article.collected_date.desc()).all()
        print(f"\nARTICLES ({len(articles)} total)")
        print("-" * 80)
        recommended_articles = [a for a in articles if a.recommended]
        print(f"Recommended: {len(recommended_articles)}")
        print(f"Not recommended: {len(articles) - len(recommended_articles)}")
        if articles:
            print("\nRecent articles:")
            for article in articles[:10]:  # Show first 10
                status = "[RECOMMENDED]" if article.recommended else "[NOT RECOMMENDED]"
                print(f"  - {article.title[:60]:60} | {status}")
        
        # Statistics
        print(f"\nSTATISTICS")
        print("-" * 80)
        interaction_counts = {}
        for interaction in interactions:
            interaction_counts[interaction.interaction_type] = interaction_counts.get(interaction.interaction_type, 0) + 1
        
        print("Interaction breakdown:")
        for interaction_type, count in sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {interaction_type:15}: {count:5}")
        
        print(f"\nTotal items in database: {len(papers) + len(articles)}")
        print(f"Total interactions: {len(interactions)}")
        
    finally:
        db.close()

if __name__ == "__main__":
    view_database()

