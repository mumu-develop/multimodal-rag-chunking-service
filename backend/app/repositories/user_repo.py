"""Repository: MySQL users data access."""
import hashlib
from typing import Optional

import structlog

from app.models import UserProfile, get_session

logger = structlog.get_logger()


class UserRepository:
    """MySQL users table repository."""

    def __init__(self):
        self.last_error: str = ""

    def create(self, username: str, password: str, role: str = "user") -> Optional[UserProfile]:
        session = get_session()
        try:
            record = UserProfile(
                username=username,
                password_hash=_hash_password(password),
                role=role,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("user_created", id=record.id, username=username)
            self.last_error = ""
            return record
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("user_create_failed", username=username, error=self.last_error)
            return None
        finally:
            session.close()

    @staticmethod
    def list(page: int = 1, size: int = 20) -> tuple[int, list[UserProfile]]:
        session = get_session()
        try:
            q = session.query(UserProfile)
            total = q.count()
            items = q.order_by(UserProfile.gmt_create.desc()) \
                     .offset((page - 1) * size).limit(size).all()
            return total, items
        finally:
            session.close()

    @staticmethod
    def get_by_id(user_id: int) -> Optional[UserProfile]:
        session = get_session()
        try:
            return session.query(UserProfile).filter(UserProfile.id == user_id).first()
        finally:
            session.close()

    @staticmethod
    def get_by_username(username: str) -> Optional[UserProfile]:
        session = get_session()
        try:
            return session.query(UserProfile).filter(UserProfile.username == username).first()
        finally:
            session.close()

    @staticmethod
    def verify_password(username: str, password: str) -> Optional[UserProfile]:
        """校验用户名密码。"""
        user = UserRepository.get_by_username(username)
        if user and user.password_hash == _hash_password(password):
            return user
        return None

    def update(self, user_id: int, **kwargs) -> bool:
        """更新用户信息。密码需要用 'password' 字段传入原文，自动哈希。"""
        if "password" in kwargs:
            kwargs["password_hash"] = _hash_password(kwargs.pop("password"))
        session = get_session()
        try:
            row = session.query(UserProfile).filter(UserProfile.id == user_id).update(kwargs)
            session.commit()
            logger.info("user_updated", id=user_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("user_update_failed", id=user_id, error=self.last_error)
            return False
        finally:
            session.close()

    def delete(self, user_id: int) -> bool:
        session = get_session()
        try:
            row = session.query(UserProfile).filter(UserProfile.id == user_id).delete()
            session.commit()
            logger.info("user_deleted", id=user_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("user_delete_failed", id=user_id, error=self.last_error)
            return False
        finally:
            session.close()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# 单例
user_repo = UserRepository()
