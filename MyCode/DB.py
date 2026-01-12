import sqlite3
import time # 用于生成默认的时间戳
from typing import Optional, Dict, Any

# 需要一个os类进行协助
import os
import sys
# ----------------------------------------------------------------------
# 模块顶层定义 BASE_DIR (确保只执行一次)
try:
    # 获取该文件所在的目录的绝对路径
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 备用方案，虽然不理想
    BASE_DIR = os.getcwd()

# Notice: 这里的 BASE_DIR 是指 code 文件夹的路径
# 在涉及到包和调库的时候，CWD 问题和python相对路径问题会变得比较复杂
# ----------------------------------------------------------------------

# 获取数据库信息，理论上这个路径在后面不会用了，因为已经改成了面向对象编程，但是先保留着以防万一
DB_FILE = '../asset/database.db'


# 数据库操作类

import os
import sqlite3
import json
from typing import Optional, List, Dict

class SQLiteDB:

    def __init__(self, db_file: str = '../asset/database.db'):
        abs_db_path = os.path.join(BASE_DIR, db_file)
        self.db_path = os.path.normpath(abs_db_path)

        print(f"📁 SQLite DB path: {self.db_path}")

    # =========================================================
    # 内部工具：获取连接（每次新建）
    # =========================================================

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # =========================================================
    # 基础工具方法
    # =========================================================

    def list_tables(self) -> List[str]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            return [row[0] for row in cursor.fetchall()]
        finally:
            conn.close()

    def describe_table(self, table_name: str) -> List[Dict]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            return [
                {
                    "cid": col[0],
                    "name": col[1],
                    "type": col[2],
                    "notnull": bool(col[3]),
                    "default": col[4],
                    "pk": bool(col[5]),
                }
                for col in columns
            ]
        finally:
            conn.close()

    # =========================================================
    # 表创建
    # =========================================================

    def create_results_table(self) -> None:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                taskID    TEXT PRIMARY KEY,
                userID    TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                results   TEXT
            )
            """)
            conn.commit()
            print("ℹ️ 表 results 已存在或创建完成")
        finally:
            conn.close()

    # =========================================================
    # 插入数据
    # =========================================================

    def insert_task(
        self,
        taskID: str,
        userID: str,
        timestamp: str,
        results: Optional[list] = None
    ) -> Optional[Dict]:

        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            results_json = json.dumps(results) if results is not None else None

            cursor.execute(
                """
                INSERT INTO results (taskID, userID, timestamp, results)
                VALUES (?, ?, ?, ?)
                """,
                (taskID, userID, timestamp, results_json)
            )

            conn.commit()
            return self.get_task_by_taskid(taskID)

        except sqlite3.IntegrityError:
            print(f"❌ taskID 已存在: {taskID}")
            return None

        finally:
            conn.close()

    # =========================================================
    # 查询方法
    # =========================================================

    def get_task_by_taskid(self, taskID: str) -> Optional[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM results WHERE taskID = ?",
                (taskID,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            record = dict(row)
            record["results"] = (
                json.loads(record["results"])
                if record["results"] is not None
                else None
            )
            return record

        finally:
            conn.close()

    def get_tasks_by_userid(self, userID: str) -> List[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM results WHERE userID = ? ORDER BY timestamp DESC",
                (userID,)
            )

            rows = cursor.fetchall()
            tasks = []

            for row in rows:
                record = dict(row)
                record["results"] = (
                    json.loads(record["results"])
                    if record["results"] is not None
                    else None
                )
                tasks.append(record)

            return tasks

        finally:
            conn.close()

    def get_all_tasks(self) -> List[Dict]:
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM results ORDER BY timestamp DESC")
            rows = cursor.fetchall()

            all_tasks = []
            for row in rows:
                record = dict(row)
                record["results"] = (
                    json.loads(record["results"])
                    if record["results"] is not None
                    else None
                )
                all_tasks.append(record)

            return all_tasks

        finally:
            conn.close()

    # =========================================================
    # 删除 / 清空方法
    # =========================================================

    def clear_results_table(self) -> None:
        """删除 results 表中的所有数据（保留表结构）"""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM results")
            conn.commit()
            print("🧹 已清空 results 表中的所有数据")
        finally:
            conn.close()