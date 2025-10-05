"""
Shared SQLAlchemy Base for all database models
"""

from sqlalchemy.ext.declarative import declarative_base

# Create a single shared base for all models
Base = declarative_base()
