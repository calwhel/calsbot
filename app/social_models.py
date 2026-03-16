"""
Social Models — Creator Profiles, Follow/Feed, Monthly Competitions
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    ForeignKey, Text, JSON, UniqueConstraint
)
from datetime import datetime

from app.database import Base


class UserFollow(Base):
    """Follower → Following relationship between portal users."""
    __tablename__ = "user_follows"

    id           = Column(Integer, primary_key=True, index=True)
    follower_id  = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    following_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at   = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("follower_id", "following_id", name="uq_follow"),)


class FeedActivity(Base):
    """
    Activity events surfaced in follower feeds.
    activity_type: strategy_published | strategy_milestone | competition_entered |
                   competition_win | strategy_updated
    """
    __tablename__ = "feed_activities"

    id            = Column(Integer, primary_key=True, index=True)
    user_id       = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    activity_type = Column(String(50), nullable=False)
    title         = Column(String(200), nullable=False)
    subtitle      = Column(String(300), nullable=True)
    strategy_id   = Column(Integer, ForeignKey("user_strategies.id"), nullable=True)
    listing_id    = Column(Integer, nullable=True)
    data          = Column(JSON, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow, index=True)


class Competition(Base):
    """Monthly paper-trading competition."""
    __tablename__ = "competitions"

    id           = Column(Integer, primary_key=True, index=True)
    title        = Column(String(140), nullable=False)
    description  = Column(Text, nullable=True)
    prize_text   = Column(String(200), nullable=True)   # e.g. "$50 USDT to winner"
    starts_at    = Column(DateTime, nullable=False)
    ends_at      = Column(DateTime, nullable=False)
    status       = Column(String(20), default="active")  # active | ended
    created_at   = Column(DateTime, default=datetime.utcnow)


class CompetitionEntry(Base):
    """A user's strategy entered into a competition."""
    __tablename__ = "competition_entries"

    id             = Column(Integer, primary_key=True, index=True)
    competition_id = Column(Integer, ForeignKey("competitions.id"), nullable=False, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    strategy_id    = Column(Integer, ForeignKey("user_strategies.id"), nullable=False)
    entered_at     = Column(DateTime, default=datetime.utcnow)
    final_rank     = Column(Integer, nullable=True)
    final_score    = Column(Float, nullable=True)

    __table_args__ = (UniqueConstraint("competition_id", "user_id", name="uq_comp_entry"),)


def init_social_tables(engine):
    """Create social tables if they don't exist. Safe to call on every startup."""
    try:
        import app.models       # noqa
        import app.strategy_models  # noqa
    except Exception:
        pass
    Base.metadata.create_all(bind=engine, tables=[
        UserFollow.__table__,
        FeedActivity.__table__,
        Competition.__table__,
        CompetitionEntry.__table__,
    ])
