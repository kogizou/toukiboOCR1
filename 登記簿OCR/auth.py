"""
登記簿OCRアプリ - ユーザー認証機能

このスクリプトは登記簿OCRアプリのユーザー認証機能を実装します。
ユーザーの登録、ログイン、ログアウト機能を提供します。
"""

import os
import sqlite3
import hashlib
import logging
from datetime import datetime
from functools import wraps
from flask import session, redirect, url_for, flash, request

# ロギングの設定
logger = logging.getLogger('RegistryOCRAuth')

class UserAuth:
    """ユーザー認証管理クラス"""
    
    def __init__(self, db_path='registry_ocr.db'):
        """
        初期化メソッド
        
        Args:
            db_path (str, optional): データベースファイルのパス。デフォルトは'registry_ocr.db'。
        """
        self.db_path = db_path
        self._create_tables()
        self._create_default_user()
        logger.info("UserAuth initialized with database: %s", db_path)
    
    def _create_tables(self):
        """ユーザーテーブルを作成する"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # ユーザーテーブルの作成
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                last_login TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            logger.info("User authentication tables created successfully")
    
    def _create_default_user(self):
        """デフォルトの管理者ユーザーを作成する"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # デフォルトユーザーが既に存在するか確認
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
                if cursor.fetchone()[0] == 0:
                    # デフォルトユーザーの作成
                    password_hash = self._hash_password('admin123')
                    cursor.execute('''
                    INSERT INTO users (username, password_hash, email, role)
                    VALUES (?, ?, ?, ?)
                    ''', ('admin', password_hash, 'admin@example.com', 'admin'))
                    
                    conn.commit()
                    logger.info("Default admin user created")
                
        except Exception as e:
            logger.error("Failed to create default user: %s", str(e), exc_info=True)
    
    def _hash_password(self, password):
        """
        パスワードをハッシュ化する
        
        Args:
            password (str): 平文パスワード
            
        Returns:
            str: ハッシュ化されたパスワード
        """
        # SHA-256ハッシュを使用
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, password, email=None, role='user'):
        """
        新しいユーザーを登録する
        
        Args:
            username (str): ユーザー名
            password (str): パスワード
            email (str, optional): メールアドレス
            role (str, optional): ユーザーロール。デフォルトは'user'。
            
        Returns:
            bool: 登録成功の場合はTrue、失敗の場合はFalse
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # ユーザー名が既に存在するか確認
                cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (username,))
                if cursor.fetchone()[0] > 0:
                    logger.warning("User registration failed: Username '%s' already exists", username)
                    return False
                
                # パスワードのハッシュ化
                password_hash = self._hash_password(password)
                
                # ユーザーの登録
                cursor.execute('''
                INSERT INTO users (username, password_hash, email, role)
                VALUES (?, ?, ?, ?)
                ''', (username, password_hash, email, role))
                
                conn.commit()
                logger.info("User '%s' registered successfully", username)
                return True
                
        except Exception as e:
            logger.error("User registration failed: %s", str(e), exc_info=True)
            return False
    
    def login(self, username, password):
        """
        ユーザーのログイン認証を行う
        
        Args:
            username (str): ユーザー名
            password (str): パスワード
            
        Returns:
            dict or None: 認証成功時はユーザー情報を含む辞書、失敗時はNone
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # ユーザーの検索
                cursor.execute('''
                SELECT id, username, email, role
                FROM users
                WHERE username = ? AND password_hash = ?
                ''', (username, self._hash_password(password)))
                
                user = cursor.fetchone()
                
                if user:
                    # 最終ログイン時間の更新
                    cursor.execute('''
                    UPDATE users SET last_login = ? WHERE id = ?
                    ''', (datetime.now().isoformat(), user['id']))
                    
                    conn.commit()
                    logger.info("User '%s' logged in successfully", username)
                    return dict(user)
                else:
                    logger.warning("Login failed for user '%s': Invalid credentials", username)
                    return None
                
        except Exception as e:
            logger.error("Login failed: %s", str(e), exc_info=True)
            return None
    
    def get_user(self, user_id):
        """
        ユーザーIDからユーザー情報を取得する
        
        Args:
            user_id (int): ユーザーID
            
        Returns:
            dict or None: ユーザー情報を含む辞書、存在しない場合はNone
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                SELECT id, username, email, role, last_login, created_at
                FROM users
                WHERE id = ?
                ''', (user_id,))
                
                user = cursor.fetchone()
                
                return dict(user) if user else None
                
        except Exception as e:
            logger.error("Failed to get user: %s", str(e), exc_info=True)
            return None
    
    def change_password(self, user_id, current_password, new_password):
        """
        ユーザーのパスワードを変更する
        
        Args:
            user_id (int): ユーザーID
            current_password (str): 現在のパスワード
            new_password (str): 新しいパスワード
            
        Returns:
            bool: 変更成功の場合はTrue、失敗の場合はFalse
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 現在のパスワードを確認
                cursor.execute('''
                SELECT COUNT(*) FROM users
                WHERE id = ? AND password_hash = ?
                ''', (user_id, self._hash_password(current_password)))
                
                if cursor.fetchone()[0] == 0:
                    logger.warning("Password change failed: Current password is incorrect")
                    return False
                
                # パスワードの更新
                cursor.execute('''
                UPDATE users SET password_hash = ? WHERE id = ?
                ''', (self._hash_password(new_password), user_id))
                
                conn.commit()
                logger.info("Password changed successfully for user ID %d", user_id)
                return True
                
        except Exception as e:
            logger.error("Password change failed: %s", str(e), exc_info=True)
            return False

# フラスク用のデコレータ関数
def login_required(f):
    """ログインが必要なルートに対するデコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('この機能を使用するにはログインが必要です', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """管理者権限が必要なルートに対するデコレータ"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('この機能を使用するにはログインが必要です', 'warning')
            return redirect(url_for('login', next=request.url))
        if session.get('user_role') != 'admin':
            flash('この機能を使用するには管理者権限が必要です', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function 