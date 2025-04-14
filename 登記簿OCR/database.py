"""
登記簿OCRアプリ - データベース連携

このスクリプトは登記簿OCRアプリのデータベース連携機能を実装します。
SQLiteデータベースを使用して、OCR処理結果の永続化と検索機能を提供します。
"""

import os
import sqlite3
import json
import logging
from datetime import datetime

# ロギングの設定
logger = logging.getLogger('RegistryOCRDB')

class OCRDatabase:
    """OCR処理結果のデータベース管理クラス"""
    
    def __init__(self, db_path='registry_ocr.db'):
        """
        初期化メソッド
        
        Args:
            db_path (str, optional): データベースファイルのパス。デフォルトは'registry_ocr.db'。
        """
        self.db_path = db_path
        self._create_tables()
        logger.info("OCRDatabase initialized with database: %s", db_path)
    
    def _create_tables(self):
        """データベースのテーブルを作成する"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 不動産テーブルの作成
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS properties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_number TEXT UNIQUE,
                address TEXT,
                lot_number TEXT,
                area REAL,
                area_unit TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 所有者テーブルの作成
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS owners (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 所有者変更履歴テーブルの作成
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ownership_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_id INTEGER,
                owner_id INTEGER,
                previous_owner_id INTEGER,
                event_type TEXT,
                event_date DATE,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (property_id) REFERENCES properties (id),
                FOREIGN KEY (owner_id) REFERENCES owners (id),
                FOREIGN KEY (previous_owner_id) REFERENCES owners (id)
            )
            ''')
            
            # OCR処理結果テーブルの作成
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS ocr_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                original_text TEXT,
                structured_data TEXT,
                confidence REAL,
                property_id INTEGER,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (property_id) REFERENCES properties (id)
            )
            ''')
            
            conn.commit()
            logger.info("Database tables created successfully")
    
    def save_ocr_result(self, result):
        """
        OCR処理結果をデータベースに保存する
        
        Args:
            result (dict): OCR処理結果を含む辞書
            
        Returns:
            int: 保存されたレコードのID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 構造化データの取得
                structured_data = result.get('structured_data', {})
                
                # 不動産情報の取得
                property_number = structured_data.get('property_number', '')
                address = structured_data.get('address', '')
                lot_number = structured_data.get('lot_number', '')
                area = structured_data.get('area', 0)
                area_unit = structured_data.get('area_unit', '㎡')
                
                # 所有者情報の取得
                owner_name = structured_data.get('owner_name', '')
                owner_address = structured_data.get('owner_address', '')
                
                # 不動産データの保存
                cursor.execute('''
                INSERT OR IGNORE INTO properties 
                (property_number, address, lot_number, area, area_unit) 
                VALUES (?, ?, ?, ?, ?)
                ''', (property_number, address, lot_number, area, area_unit))
                
                # 不動産IDの取得
                cursor.execute('SELECT id FROM properties WHERE property_number = ?', (property_number,))
                property_row = cursor.fetchone()
                property_id = property_row['id'] if property_row else None
                
                # 所有者データの保存
                if owner_name:
                    cursor.execute('''
                    INSERT INTO owners (name, address) VALUES (?, ?)
                    ''', (owner_name, owner_address))
                    owner_id = cursor.lastrowid
                    
                    # 所有者変更履歴の保存
                    if property_id:
                        # 前の所有者を確認
                        cursor.execute('''
                        SELECT owner_id FROM ownership_history 
                        WHERE property_id = ? 
                        ORDER BY event_date DESC LIMIT 1
                        ''', (property_id,))
                        prev_owner_row = cursor.fetchone()
                        previous_owner_id = prev_owner_row['owner_id'] if prev_owner_row else None
                        
                        # 変更履歴を追加
                        event_date = datetime.now().strftime('%Y-%m-%d')
                        cursor.execute('''
                        INSERT INTO ownership_history 
                        (property_id, owner_id, previous_owner_id, event_type, event_date) 
                        VALUES (?, ?, ?, ?, ?)
                        ''', (property_id, owner_id, previous_owner_id, '所有権移転', event_date))
                
                # OCR結果の保存
                cursor.execute('''
                INSERT INTO ocr_results 
                (file_name, original_text, structured_data, confidence, property_id, processed_at) 
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    result.get('file_name', ''),
                    result.get('text', ''),
                    json.dumps(structured_data, ensure_ascii=False),
                    result.get('confidence', 0),
                    property_id,
                    result.get('processed_at', datetime.now().isoformat())
                ))
                
                result_id = cursor.lastrowid
                conn.commit()
                logger.info("OCR result saved to database with ID: %d", result_id)
                return result_id
                
        except Exception as e:
            logger.error("Failed to save OCR result to database: %s", str(e), exc_info=True)
            return None
    
    def get_all_properties(self):
        """
        すべての不動産情報を取得する
        
        Returns:
            list: 不動産情報のリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT p.*, o.name as current_owner
                FROM properties p
                LEFT JOIN (
                    SELECT property_id, owner_id
                    FROM ownership_history
                    WHERE (property_id, event_date) IN (
                        SELECT property_id, MAX(event_date)
                        FROM ownership_history
                        GROUP BY property_id
                    )
                ) h ON p.id = h.property_id
                LEFT JOIN owners o ON h.owner_id = o.id
                ORDER BY p.created_at DESC
                ''')
                
                rows = cursor.fetchall()
                properties = []
                for row in rows:
                    properties.append(dict(row))
                
                return properties
                
        except Exception as e:
            logger.error("Failed to get properties: %s", str(e), exc_info=True)
            return []
    
    def get_all_owners(self):
        """
        すべての所有者情報を取得する
        
        Returns:
            list: 所有者情報のリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT o.*, COUNT(DISTINCT h.property_id) as property_count
                FROM owners o
                LEFT JOIN ownership_history h ON o.id = h.owner_id
                GROUP BY o.id
                ORDER BY o.name
                ''')
                
                rows = cursor.fetchall()
                owners = []
                for row in rows:
                    owners.append(dict(row))
                
                return owners
                
        except Exception as e:
            logger.error("Failed to get owners: %s", str(e), exc_info=True)
            return []
    
    def get_property_timeline(self, property_id):
        """
        不動産の所有者変更履歴を取得する
        
        Args:
            property_id (int): 不動産ID
            
        Returns:
            list: 所有者変更履歴のリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT h.*, 
                       o.name as owner_name, 
                       p.name as previous_owner_name
                FROM ownership_history h
                JOIN owners o ON h.owner_id = o.id
                LEFT JOIN owners p ON h.previous_owner_id = p.id
                WHERE h.property_id = ?
                ORDER BY h.event_date DESC
                ''', (property_id,))
                
                rows = cursor.fetchall()
                timeline = []
                for row in rows:
                    timeline.append(dict(row))
                
                return timeline
                
        except Exception as e:
            logger.error("Failed to get property timeline: %s", str(e), exc_info=True)
            return []
    
    def search_data(self, query):
        """
        データの検索を行う
        
        Args:
            query (str): 検索クエリ
            
        Returns:
            dict: 検索結果（プロパティと所有者のリスト）
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # 検索クエリをワイルドカード検索用に整形
                search_term = f'%{query}%'
                
                # 不動産の検索
                cursor.execute('''
                SELECT p.*, o.name as current_owner
                FROM properties p
                LEFT JOIN (
                    SELECT property_id, owner_id
                    FROM ownership_history
                    WHERE (property_id, event_date) IN (
                        SELECT property_id, MAX(event_date)
                        FROM ownership_history
                        GROUP BY property_id
                    )
                ) h ON p.id = h.property_id
                LEFT JOIN owners o ON h.owner_id = o.id
                WHERE p.property_number LIKE ? 
                OR p.address LIKE ? 
                OR p.lot_number LIKE ?
                ''', (search_term, search_term, search_term))
                
                property_rows = cursor.fetchall()
                properties = [dict(row) for row in property_rows]
                
                # 所有者の検索
                cursor.execute('''
                SELECT o.*, COUNT(DISTINCT h.property_id) as property_count
                FROM owners o
                LEFT JOIN ownership_history h ON o.id = h.owner_id
                WHERE o.name LIKE ? OR o.address LIKE ?
                GROUP BY o.id
                ''', (search_term, search_term))
                
                owner_rows = cursor.fetchall()
                owners = [dict(row) for row in owner_rows]
                
                return {
                    'properties': properties,
                    'owners': owners
                }
                
        except Exception as e:
            logger.error("Failed to search data: %s", str(e), exc_info=True)
            return {'properties': [], 'owners': []}
    
    def delete_result(self, result_id):
        """
        OCR処理結果を削除する
        
        Args:
            result_id (int): 削除するレコードのID
            
        Returns:
            bool: 削除成功の場合はTrue、失敗の場合はFalse
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM ocr_results WHERE id = ?', (result_id,))
                conn.commit()
                logger.info("OCR result deleted with ID: %d", result_id)
                return True
                
        except Exception as e:
            logger.error("Failed to delete OCR result: %s", str(e), exc_info=True)
            return False 