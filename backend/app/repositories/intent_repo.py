"""Repository: MySQL intent_tree_nodes data access."""
from typing import Optional

import structlog

from app.models import IntentTreeNode, get_session

logger = structlog.get_logger()


class IntentTreeRepository:
    """MySQL intent_tree_nodes table repository."""

    def __init__(self):
        self.last_error: str = ""

    def create_node(self, name: str, node_type: str, parent_id: Optional[int] = None,
                    sort_order: int = 0, collection_name: str = "",
                    description: str = "", example_questions: list[str] | None = None) -> Optional[IntentTreeNode]:
        session = get_session()
        try:
            record = IntentTreeNode(
                name=name,
                node_type=node_type,
                parent_id=parent_id,
                sort_order=sort_order,
                collection_name=collection_name or None,
                description=description or None,
                example_questions=example_questions or [],
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            logger.info("intent_node_created", id=record.id, name=name, node_type=node_type)
            self.last_error = ""
            return record
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("intent_node_create_failed", name=name, error=self.last_error)
            return None
        finally:
            session.close()

    @staticmethod
    def get_tree() -> list[dict]:
        """获取整棵意图树，按 parent_id + sort_order 排序。"""
        session = get_session()
        try:
            nodes = session.query(IntentTreeNode) \
                .order_by(IntentTreeNode.parent_id.is_(None).desc(),
                          IntentTreeNode.parent_id,
                          IntentTreeNode.sort_order) \
                .all()
            return [_node_to_dict(n) for n in nodes]
        finally:
            session.close()

    @staticmethod
    def get_by_id(node_id: int) -> Optional[IntentTreeNode]:
        session = get_session()
        try:
            return session.query(IntentTreeNode).filter(IntentTreeNode.id == node_id).first()
        finally:
            session.close()

    def update_node(self, node_id: int, **kwargs) -> bool:
        session = get_session()
        try:
            row = session.query(IntentTreeNode).filter(IntentTreeNode.id == node_id).update(kwargs)
            session.commit()
            logger.info("intent_node_updated", id=node_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("intent_node_update_failed", id=node_id, error=self.last_error)
            return False
        finally:
            session.close()

    def delete_node(self, node_id: int) -> bool:
        """删除节点（需先确保无子节点）。"""
        session = get_session()
        try:
            children = session.query(IntentTreeNode) \
                .filter(IntentTreeNode.parent_id == node_id).count()
            if children > 0:
                self.last_error = f"节点有 {children} 个子节点，无法删除"
                logger.warning("intent_node_has_children", id=node_id, children=children)
                return False
            row = session.query(IntentTreeNode).filter(IntentTreeNode.id == node_id).delete()
            session.commit()
            logger.info("intent_node_deleted", id=node_id, row_count=row)
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("intent_node_delete_failed", id=node_id, error=self.last_error)
            return False
        finally:
            session.close()

    def toggle_node(self, node_id: int, is_enabled: bool) -> bool:
        session = get_session()
        try:
            row = session.query(IntentTreeNode).filter(IntentTreeNode.id == node_id) \
                .update({"is_enabled": 1 if is_enabled else 0})
            session.commit()
            self.last_error = ""
            return row > 0
        except Exception as e:
            session.rollback()
            self.last_error = str(e)
            logger.exception("intent_node_toggle_failed", id=node_id, error=self.last_error)
            return False
        finally:
            session.close()

    @staticmethod
    def get_enabled_topic_nodes() -> list[dict]:
        """获取所有启用的 TOPIC 类型节点（用于意图识别）。"""
        session = get_session()
        try:
            nodes = session.query(IntentTreeNode) \
                .filter(IntentTreeNode.node_type == "TOPIC") \
                .filter(IntentTreeNode.is_enabled == 1) \
                .filter(IntentTreeNode.collection_name.isnot(None)) \
                .filter(IntentTreeNode.collection_name != "") \
                .order_by(IntentTreeNode.sort_order) \
                .all()
            return [_node_to_dict(n) for n in nodes]
        finally:
            session.close()


def _node_to_dict(node: IntentTreeNode) -> dict:
    return {
        "id": node.id,
        "name": node.name,
        "node_type": node.node_type,
        "parent_id": node.parent_id,
        "sort_order": node.sort_order,
        "collection_name": node.collection_name,
        "description": node.description or "",
        "example_questions": node.example_questions or [],
        "is_enabled": node.is_enabled,
        "gmt_create": node.gmt_create.isoformat() if node.gmt_create else "",
        "gmt_modified": node.gmt_modified.isoformat() if node.gmt_modified else "",
    }


# 单例
intent_repo = IntentTreeRepository()
