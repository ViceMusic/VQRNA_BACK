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

class SQLiteDB:
    def __init__(self, db_file: str='../asset/database.db'):   # 路径应该是没有问题的才对
        # 1. 使用 BASE_DIR 拼接传入的相对路径
        abs_db_path = os.path.join(BASE_DIR, db_file)
        final_db_path = os.path.normpath(abs_db_path)
        
        self.conn = None
        try:
            print(f"正在连接/创建数据库文件：{final_db_path} ...")
            # 🌟 使用绝对路径连接
            self.conn = sqlite3.connect(final_db_path) 
            print(f"✅ 成功连接/创建数据库文件：{final_db_path}")
        except sqlite3.Error as e:
            print(f"❌ 连接数据库时发生错误: {e}")
            # 推荐抛出异常，阻止程序继续运行
            raise RuntimeError(f"致命错误：无法打开数据库文件: {final_db_path}") from e

    def connect(self):
        print("该方法只做测试查看是否调用")
    
    def __exit__(self):  # 自动对数据库进行回收操作，说实话我简直是在赌
        if self.conn:
            self.conn.close()

    # --这些是操作数据库的通用函数--
    
    #=========操作区域=================
    # ---获取表中所有数据，并以字典列表形式返回---
    # 传入参数为表名，如果查询得到数据会返回一个字典列表，否则返回空列表
    # 这个方法只在测试阶段使用，用于查看所有的表结构，正式环境请使用fetch_by_userid_as_dicts
    def fetch_all_as_dicts(self,table_name: str) :
        conn = self.conn
        results = []
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            sql_query = f"SELECT * FROM {table_name}" # 构造 SELECT * 语句,获取执行结果储存在cursor中
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            # 遍历每一行 sqlite3.Row 对象，并将其转换为 Python 字典
            for row in rows:
                results.append(dict(row)) # dict(row) 会将 sqlite3.Row 对象（行为类似字典）转换为真正的字典
            
        except sqlite3.Error as e:
            print(e)

            return [] # 返回空列表表示失败
                
        return results

    # ---获取表中特定userid的任务，并以字典列表形式返回---
    # 传入参数为表名和userid，如果查询得到数据会返回一个字典列表，否则返回空列表
    def fetch_by_userid_as_dicts(self,table_name: str, userid: str) :
        conn = self.conn
        results = []
        
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            sql_query = f"SELECT * FROM {table_name} WHERE userid = '{userid}'" # 构造 SELECT * 语句,获取执行结果储存在cursor中
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            # 遍历每一行 sqlite3.Row 对象，并将其转换为 Python 字典
            for row in rows:
                results.append(dict(row)) # dict(row) 会将 sqlite3.Row 对象（行为类似字典）转换为真正的字典
            
        except sqlite3.Error as e:
            print(e)
            return [] # 返回空列表表示失败
        
        return results

    # ---插入新数据，并且这个插入的数据本身以dist返回---
    # items 允许为空 updated_time: Optional[str] = None # updated_time 允许传入，否则自动生成、
    # 如果插入成果，会返回一个dict，否则返回None
    def insert_motif_record(self,taskid: str,  userid: str,  pending: int,  items: Optional[str] = None, updated_time: Optional[str] = None) -> dict:
        """
        插入一条新的 motif 记录到数据库，并返回新插入的记录（字典格式）。

        parameters:
        taskid: 任务 ID (TEXT NOT NULL)
        userid: 用户 ID (TEXT NOT NULL)
        pending: 状态值 (INTEGER NOT NULL, 0 或 1)
        items: JSON 格式的字符串，可为空 (TEXT)
        updated_time: 记录更新时间，默认为当前时间 (TEXT NOT NULL)

        return: 
        插入成功的记录字典，失败则返回 None
        """
        conn = self.conn
        
        # 如果没有提供时间，则使用当前系统时间
        if updated_time is None:
            updated_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        try:
            cursor = conn.cursor()

            # 1. 定义 SQL 插入语句，插入任务id、用户id、状态、items 和更新时间
            sql_insert = """
            INSERT INTO motif (taskid, userid, state, items, update_time)
            VALUES (?, ?, ?, ?, ?)
            """
            
            data_to_insert = (taskid, userid, pending, items, updated_time)
            
            # 2. 执行插入操作
            cursor.execute(sql_insert, data_to_insert)
            
            # 3. 获取新插入记录的自增 ID
            new_id = cursor.lastrowid
            
            # 4. 提交事务，确保数据写入数据库
            conn.commit()
            print(f"✅ 记录插入成功，新 ID 为: {new_id}")

            # 5. 查询新插入的记录并返回
            # 再次设置 row_factory 以便将结果转换为字典
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 根据获取到的 ID 查询该记录的所有字段
            cursor.execute("SELECT * FROM motif WHERE id = ?", (new_id,))
            
            # 获取结果行
            row = cursor.fetchone()
            
            if row:
                return dict(row)# 转换为字典并返回
            else:
                return None

        except sqlite3.Error as e:
            print(f"❌ 插入数据库时发生错误: {e}")
            return None 
            
       

    # ---更新 motif 表中的 items 字段---
    # 传入参数为任务id 和新的 items 字段内容（JSON字符串）
    def update_motif_items_by_taskid(self,taskid: str, state: int, new_items: str) -> dict:
        """
        更新 motif 表中指定 taskid 的记录的 items 和 state 字段，并返回更新后的记录。

        :param taskid: 要查找更新的任务 ID。
        :param state: 要设置的新状态值 (0, 1, 2)。
        :param new_items: JSON 格式的字符串，作为 items 字段的新内容。
        :return: 更新后的记录字典，如果 taskid 不存在或更新失败，则返回 None。
        """
        conn = self.conn
        
        # 1. 校验 state 参数并修正 要求 state 如果不是 0, 1, 或 2，则自动改编为 0
        valid_states = {0, 1, 2}
        if state not in valid_states:
            print(f"⚠️ 状态值 {state} 无效，已自动设置为默认值 0。")
            state = 0
            
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())     # 获取当前时间作为更新时间

        try:
            cursor = conn.cursor()

            # 2. 检查 taskid 是否存在 (防止不必要的 UPDATE)
            cursor.execute("SELECT id FROM motif WHERE taskid = ?", (taskid,))
            if cursor.fetchone() is None:
                print(f"❌ 记录更新失败：taskid '{taskid}' 不存在于数据库中。")
                return None

            sql_update = """
            UPDATE motif
            SET items = ?, state = ?, update_time = ?
            WHERE taskid = ?
            """
            
            data_to_update = (new_items, state, current_time, taskid)
            
            # 3 执行更新操作
            cursor.execute(sql_update, data_to_update)
            
            # 4. 提交事务
            conn.commit()
            print(f"✅ 记录更新成功，TaskID: {taskid}，State: {state}。")

            # 5. 查询并返回更新后的记录
            
            # 设置 row_factory 以便将结果转换为字典
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 根据 taskid 查询更新后的记录
            cursor.execute("SELECT * FROM motif WHERE taskid = ?", (taskid,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            else:
                print("❌ 更新成功但无法检索更新后的记录。")
                return None

        except sqlite3.Error as e:
            print(f"❌ 数据库操作发生错误: {e}")
            return None 
            


    # --删除motif表中指定taskid的记录---
    def delete_motif_record_by_taskid(self,taskid: str) -> dict:
        """
        根据 taskid 删除 motif 表中的记录。

        :param taskid: 要删除记录的任务 ID。
        :return: 包含删除结果和删除数量的字典。
        """
        conn = self.conn
        deleted_count = 0
        
        try:
            cursor = conn.cursor()

            # 1. 定义 SQL DELETE 语句
            sql_delete = "DELETE FROM motif WHERE taskid = ?"
            
            # 2. 执行删除操作
            cursor.execute(sql_delete, (taskid,))
            
            # 3. 获取被删除的行数
            deleted_count = cursor.rowcount
            
            # 4. 提交事务
            conn.commit()
            
            if deleted_count > 0:
                message = f"✅ 成功删除 {deleted_count} 条 taskid 为 '{taskid}' 的记录。"
                print(message)
            else:
                message = f"⚠️ 没有找到 taskid 为 '{taskid}' 的记录，未执行删除操作。"
                print(message)

            return {
                "taskid": taskid,
                "deleted_count": deleted_count,
                "success": deleted_count > 0,
                "message": message
            }

        except sqlite3.Error as e:
            error_message = f"❌ 删除 taskid 记录时发生错误: {e}"
            print(error_message)
            return {
                "taskid": taskid,
                "deleted_count": 0,
                "success": False,
                "message": error_message
            }
            


    # --删除motif表中指定userid的所有记录---
    def delete_motif_record_by_userid(self,userid: str) -> dict:
        """
        根据 taskid 删除 motif 表中的记录。

        :param taskid: 要删除记录的任务 ID。
        :return: 包含删除结果和删除数量的字典。
        """
        conn = self.conn
        deleted_count = 0
        
        try:
            cursor = conn.cursor()

            # 1. 定义 SQL DELETE 语句
            sql_delete = "DELETE FROM motif WHERE userid = ?"
            
            # 2. 执行删除操作
            cursor.execute(sql_delete, (userid,))
            
            # 3. 获取被删除的行数
            deleted_count = cursor.rowcount
            
            # 4. 提交事务
            conn.commit()
            
            if deleted_count > 0:
                message = f"✅ 成功删除 {deleted_count} 条 userid 为 '{userid}' 的记录。"
                print(message)
            else:
                message = f"⚠️ 没有找到 userid 为 '{userid}' 的记录，未执行删除操作。"
                print(message)

            return {
                "userid": userid,
                "deleted_count": deleted_count,
                "success": deleted_count > 0,
                "message": message
            }

        except sqlite3.Error as e:
            error_message = f"❌ 删除 userid 记录时发生错误: {e}"
            print(error_message)
            return {
                "userid": userid,
                "deleted_count": 0,
                "success": False,
                "message": error_message
            }
            

