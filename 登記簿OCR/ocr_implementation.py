"""
登記簿OCRアプリ - OCR機能実装

このスクリプトは登記簿OCRアプリのコアとなるOCR機能を実装します。
テッセラクトOCRエンジンを使用して、登記簿のスキャン画像からテキストを抽出し、
構造化データに変換する機能を提供します。
"""

import os
import sys
import cv2
import numpy as np
import pytesseract
from PIL import Image
import pandas as pd
import re
import json
from datetime import datetime
import logging
# PDF対応のためのライブラリを追加
import tempfile
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyMuPDF not installed. PDF support will be disabled.")

# Tesseractインストール確認
try:
    # テスト実行
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception as e:
    TESSERACT_AVAILABLE = False
    logging.warning(f"Tesseract not available: {str(e)}. OCR functionality will be limited.")

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ocr_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RegistryOCR')

class RegistryOCR:
    """登記簿OCR処理のメインクラス"""
    
    def __init__(self, config=None):
        """
        初期化メソッド
        
        Args:
            config (dict, optional): 設定情報を含む辞書。デフォルトはNone。
        """
        # デフォルト設定
        self.default_config = {
            'lang': 'jpn',  # 日本語
            'tesseract_cmd': 'tesseract',  # Tesseractコマンドのパス
            'temp_dir': './temp',  # 一時ファイル保存ディレクトリ
            'output_dir': './output',  # 出力ディレクトリ
            'confidence_threshold': 70,  # 信頼度閾値（%）
            'dpi': 300,  # DPI設定
            'psm': 6,  # ページセグメンテーションモード
            'oem': 3,  # OCRエンジンモード
        }
        
        # 設定の初期化
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Tesseractのパス設定
        # Mac環境でのTesseractパスの可能性
        mac_tesseract_paths = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract',
            # ユーザーインストールパス
            os.path.expanduser('~/bin/tesseract'),
            # アプリケーションパス
            '/Applications/Tesseract.app/Contents/MacOS/tesseract'
        ]
        
        # 環境に応じたパス設定
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
        elif os.name == 'posix':  # MacやLinux
            # 既存のパスが有効かチェック
            if os.path.isfile(self.config['tesseract_cmd']) and os.access(self.config['tesseract_cmd'], os.X_OK):
                pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
            else:
                # Mac環境での可能性のあるパスをチェック
                for path in mac_tesseract_paths:
                    if os.path.isfile(path) and os.access(path, os.X_OK):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self.config['tesseract_cmd'] = path
                        logging.info(f"Found Tesseract at: {path}")
                        break
        
        # 一時ディレクトリと出力ディレクトリの作成
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # 登記簿の構造定義（正規表現パターン）
        self.patterns = {
            'property_number': r'不動産番号[：:]\s*(\d{4}[-－]\d{4}[-－]\d{4})',
            'address': r'所\s*在\s*(.+[都道府県].+[市区町村].+)',
            'lot_number': r'地\s*番\s*(.+)',
            'area': r'地\s*積\s*([0-9０-９,.，．]+)\s*([平方メートル㎡])',
            'owner_name': r'名\s*義\s*人\s*(.+)',
            'owner_address': r'住\s*所\s*(.+)',
            'rights': r'権\s*利\s*者\s*(.+)'
        }
        
        logger.info("RegistryOCR initialized with config: %s", self.config)
    
    def preprocess_image(self, image_path):
        """
        画像の前処理を行う
        
        Args:
            image_path (str): 処理する画像のパス
            
        Returns:
            numpy.ndarray: 前処理された画像
        """
        logger.info("Processing image: %s", image_path)
        
        # PDFファイルの処理
        if image_path.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                raise ValueError("PDFサポートが無効です。PyMuPDFライブラリをインストールしてください。")
            
            try:
                # PDFから画像に変換
                logger.info("Converting PDF to image: %s", image_path)
                temp_dir = tempfile.mkdtemp(dir=self.config['temp_dir'])
                
                # PyMuPDFでPDFを開く
                pdf_document = fitz.open(image_path)
                
                if len(pdf_document) == 0:
                    raise ValueError(f"PDFにページが含まれていません: {image_path}")
                
                # 最初のページを取得（インデックスは0から始まる）
                pdf_page = pdf_document[0]
                
                # ページを画像として取得 (高解像度用に設定)
                zoom_factor = self.config['dpi'] / 72 * 1.5  # 高解像度化（1.5倍）
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = pdf_page.get_pixmap(matrix=matrix, alpha=False)
                
                # 一時画像ファイルとして保存
                temp_image_path = os.path.join(temp_dir, "pdf_page_1.png")
                pix.save(temp_image_path)
                logger.info("PDF converted to image: %s", temp_image_path)
                
                # 変換された画像を読み込み
                image = cv2.imread(temp_image_path)
                if image is None:
                    raise ValueError(f"変換されたPDF画像を読み込めませんでした: {temp_image_path}")
                
                # 使用後にPDFを閉じる
                pdf_document.close()
            except Exception as e:
                logger.error("PDF変換中にエラーが発生しました: %s", str(e), exc_info=True)
                raise ValueError(f"PDF変換中にエラーが発生しました: {str(e)}")
        else:
            # 通常の画像ファイルの処理
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"画像を読み込めませんでした: {image_path}")
        
        # 画像のDPI調整
        image_height, image_width = image.shape[:2]
        current_dpi = max(image_width, image_height) / 8.5  # A4用紙を想定
        scale_factor = self.config['dpi'] / current_dpi if current_dpi > 0 else 1
        
        if scale_factor != 1:
            logger.info("Resizing image with scale factor: %f", scale_factor)
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        
        # デバッグ用：元画像を保存
        original_debug_path = os.path.join(self.config['temp_dir'], 'debug_original.png')
        cv2.imwrite(original_debug_path, image)
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 傾き検出を改善（特に縦書きの日本語テキスト向け）
        try:
            # より堅牢な傾き検出のためHough変換を使用
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 - x1 != 0:  # 垂直線を避ける
                        angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                        angles.append(angle)
                
                if angles:
                    # 最も頻度の高い角度を取得（ヒストグラムのピーク）
                    angle_counts = {}
                    for angle in angles:
                        # 0.5度単位で丸める
                        rounded = round(angle * 2) / 2
                        if rounded in angle_counts:
                            angle_counts[rounded] += 1
                        else:
                            angle_counts[rounded] = 1
                    
                    # 最頻値の角度を取得
                    angle = max(angle_counts.items(), key=lambda x: x[1])[0]
                    
                    if abs(angle) > 0.5:  # 0.5度以上の傾きがある場合のみ補正
                        logger.info("Correcting image skew using Hough transform: %f degrees", angle)
                        (h, w) = gray.shape[:2]
                        center = (w // 2, h // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                # Hough変換が失敗した場合、従来の方法を試す
                coords = np.column_stack(np.where(gray > 0))
                angle = cv2.minAreaRect(coords)[-1]
                
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                    
                if abs(angle) > 0.5:  # 0.5度以上の傾きがある場合のみ補正
                    logger.info("Correcting image skew using minAreaRect: %f degrees", angle)
                    (h, w) = gray.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.warning("Failed to correct image skew: %s", str(e))
        
        # コントラスト強調（アダプティブヒストグラム平坦化）
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # ノイズ除去の強化
        # まず、バイラテラルフィルタでエッジを保存しながらノイズ除去
        gray = cv2.bilateralFilter(gray, 11, 75, 75)
        
        # 適応的二値化（局所的な照明条件に対応）
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 15, 2)
        
        # 反転して通常の二値化に戻す（テキストが黒、背景が白）
        binary = cv2.bitwise_not(binary)
        
        # モルフォロジー演算でノイズ除去と文字の補強
        # 小さなノイズを除去
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 文字の連結部分を補強
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 登記簿特有の縦線・横線を除去
        # 水平方向のノイズ（横線）の検出と除去
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        binary = cv2.subtract(binary, horizontal_lines)
        
        # 垂直方向のノイズ（縦線）の検出と除去
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        binary = cv2.subtract(binary, vertical_lines)
        
        # エッジ検出と強調を追加して文字の境界を明確に
        edges = cv2.Canny(gray, 100, 200)
        binary = cv2.bitwise_or(binary, edges)
        
        # 結果を一時ファイルとして保存（デバッグ用）
        debug_path = os.path.join(self.config['temp_dir'], 'debug_preprocess.png')
        cv2.imwrite(debug_path, binary)
        logger.info("Preprocessed image saved to: %s", debug_path)
        
        # 登記簿で特に重要な区域を検出するために別のデバッグ画像も生成
        debug_clahe_path = os.path.join(self.config['temp_dir'], 'debug_clahe.png')
        cv2.imwrite(debug_clahe_path, gray)
        
        debug_edges_path = os.path.join(self.config['temp_dir'], 'debug_edges.png')
        cv2.imwrite(debug_edges_path, edges)
        
        return binary
    
    def detect_table_structure(self, image):
        """
        画像内のテーブル構造を検出する
        
        Args:
            image (numpy.ndarray): 処理する画像
            
        Returns:
            list: 検出されたテーブルの座標リスト
        """
        # 水平・垂直線の検出
        horizontal = np.copy(image)
        vertical = np.copy(image)
        
        # 画像の幅と高さを取得
        img_height, img_width = horizontal.shape
        
        # 水平線の検出
        horizontal_size = int(img_width / 30)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure)
        
        # 垂直線の検出
        vertical_size = int(img_height / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        vertical = cv2.erode(vertical, verticalStructure)
        vertical = cv2.dilate(vertical, verticalStructure)
        
        # 水平線と垂直線の交点を検出
        mask = horizontal + vertical
        
        # 輪郭検出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # テーブルセルの座標を取得
        table_cells = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # 小さすぎるセルは除外
                table_cells.append((x, y, w, h))
        
        return table_cells
    
    def extract_text(self, image, lang='jpn'):
        """
        画像からテキストを抽出する
        
        Args:
            image (numpy.ndarray): 処理する画像
            lang (str, optional): OCR言語。デフォルトは'jpn'。
            
        Returns:
            str: 抽出されたテキスト
        """
        # Tesseractが利用可能かチェック
        if not TESSERACT_AVAILABLE:
            logging.warning("Tesseract not available. Returning placeholder text.")
            # Tesseractがない場合、プレースホルダーテキストを返す
            return """
            Tesseractがインストールされていないため、OCR処理ができません。
            Tesseractをインストールするか、別の方法でテキスト抽出を行ってください。
            
            Tesseractのインストール方法:
            - MacOS: brew install tesseract tesseract-lang
            - Windows: https://github.com/UB-Mannheim/tesseract/wiki からインストーラーをダウンロード
            - Linux: apt-get install tesseract-ocr libtesseract-dev
            
            PDFの変換は正常に完了しています。
            """
        
        # PILイメージに変換
        pil_image = Image.fromarray(image)
        
        # 登記簿向けの最適化設定
        # --psm 6: 単一のテキストブロックとして処理（デフォルト）
        # --psm 4: 縦書きテキストの処理が必要な場合
        # --psm 3: 複雑なレイアウトで自動ページセグメンテーション
        # --oem 3: LSTMエンジンのみ（精度優先）
        # -c preserve_interword_spaces=1: 単語間のスペースを保持
        # -c tessedit_char_whitelist: 特定の文字セットに限定する場合
        
        # 登記簿OCRに最適化した設定
        config = (
            f'--psm {self.config.get("psm", 6)} '
            f'--oem {self.config.get("oem", 3)} '
            f'-c preserve_interword_spaces=1 '
            f'-c tessedit_do_invert=0 '  # 反転しない
            f'-c textord_tabfind_find_tables=1 '  # テーブル検出
            f'-c textord_tablefind_recognize_tables=1 '  # テーブル認識
            f'-c language_model_ngram_on=1 '  # 言語モデル使用
            f'-c textord_heavy_nr=1 '  # 重いノイズ除去
            f'-c tessedit_create_hocr=1 '  # HOCRファイルを作成
        )
        
        # 縦書きテキスト検出のテスト
        # 縦書きと横書きの両方でテストし、信頼度が高い方を採用
        try:
            # 標準（横書き）設定で抽出
            text_h = pytesseract.image_to_string(pil_image, lang=lang, config=config)
            
            # 縦書き設定で抽出
            config_v = config + ' --psm 5 -c textord_tablefind_vertical_text=1'
            text_v = pytesseract.image_to_string(pil_image, lang=lang, config=config_v)
            
            # 信頼度データを取得
            data_h = pytesseract.image_to_data(pil_image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            data_v = pytesseract.image_to_data(pil_image, lang=lang, config=config_v, output_type=pytesseract.Output.DICT)
            
            # 平均信頼度を計算
            conf_h = sum(int(x) for x in data_h['conf'] if x != '-1') / len([x for x in data_h['conf'] if x != '-1']) if [x for x in data_h['conf'] if x != '-1'] else 0
            conf_v = sum(int(x) for x in data_v['conf'] if x != '-1') / len([x for x in data_v['conf'] if x != '-1']) if [x for x in data_v['conf'] if x != '-1'] else 0
            
            logger.info(f"OCR confidence: horizontal={conf_h:.1f}%, vertical={conf_v:.1f}%")
            
            # 信頼度が高い方を選択
            if conf_v > conf_h:
                logger.info("Using vertical text orientation (higher confidence)")
                text = text_v
            else:
                logger.info("Using horizontal text orientation (higher confidence)")
                text = text_h
        except Exception as e:
            logger.warning(f"Error during orientation detection: {str(e)}. Using default orientation.")
            # エラーが発生した場合はデフォルト設定で抽出
            text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        
        # 文字列の正規化
        text = self._normalize_text(text)
        
        logger.info("Extracted text length: %d characters", len(text))
        if len(text) < 50:  # 短すぎるテキストはログに出力（デバッグ用）
            logger.warning("Extracted text is too short: %s", text)
        
        return text
    
    def _normalize_text(self, text):
        """
        抽出されたテキストを正規化する
        
        Args:
            text (str): OCRで抽出されたテキスト
            
        Returns:
            str: 正規化されたテキスト
        """
        # 全角数字を半角に変換
        zen_to_han = str.maketrans('０１２３４５６７８９．，', '0123456789.,')
        text = text.translate(zen_to_han)
        
        # 改行の正規化
        text = re.sub(r'\r\n', '\n', text)
        
        # 連続する空白の削除
        text = re.sub(r' +', ' ', text)
        
        # 特殊文字の置換
        text = text.replace('−', '-').replace('ー', '-').replace('一', '').replace('‐', '-')
        
        # 登記簿特有の文字パターンの補正
        # 「号」と「3」の混同修正
        text = re.sub(r'([0-9]+)号', r'\1号', text)
        text = re.sub(r'([0-9]+)3', r'\1号', text)
        
        # 「番」と「香」の混同修正
        text = re.sub(r'([0-9]+)香', r'\1番', text)
        
        # 「㎡」と「m2」の統一
        text = re.sub(r'm2', '㎡', text)
        text = re.sub(r'm²', '㎡', text)
        
        # 日付表記の統一 (例: H30.1.1 → 平成30年1月1日)
        text = re.sub(r'H([0-9]+)\.([0-9]+)\.([0-9]+)', r'平成\1年\2月\3日', text)
        text = re.sub(r'R([0-9]+)\.([0-9]+)\.([0-9]+)', r'令和\1年\2月\3日', text)
        
        # 余分な記号の削除
        text = re.sub(r'[|｜]', '', text)
        
        return text
    
    def extract_structured_data(self, text):
        """
        抽出されたテキストから構造化データを抽出する
        
        Args:
            text (str): OCRで抽出されたテキスト
            
        Returns:
            dict: 構造化データを含む辞書
        """
        # 空の構造化データ辞書を初期化
        structured_data = {
            'property_number': '',
            'address': '',
            'lot_number': '',
            'area': '',
            'area_unit': '㎡',
            'owner': {
                'name': '',
                'address': ''
            },
            'rights': []
        }
        
        # 不動産番号の抽出
        property_number_match = re.search(self.patterns['property_number'], text)
        if property_number_match:
            structured_data['property_number'] = property_number_match.group(1).strip()
        
        # 所在地の抽出（住所パターンを改善）
        address_match = re.search(self.patterns['address'], text)
        if address_match:
            structured_data['address'] = address_match.group(1).strip()
        else:
            # 別の表現パターンで再試行
            address_pattern2 = r'(?:所在|住所)\s*[:：]\s*(.+[都道府県].+[市区町村].+)'
            address_match2 = re.search(address_pattern2, text)
            if address_match2:
                structured_data['address'] = address_match2.group(1).strip()
        
        # 地番の抽出（パターンの改善）
        lot_number_match = re.search(self.patterns['lot_number'], text)
        if lot_number_match:
            structured_data['lot_number'] = lot_number_match.group(1).strip()
        else:
            # 別の表現パターンで再試行
            lot_pattern2 = r'(?:地番|番地)\s*[:：]\s*([0-9０-９-－]+(?:[番号][地目][0-9０-９-－]*)?)'
            lot_match2 = re.search(lot_pattern2, text)
            if lot_match2:
                structured_data['lot_number'] = lot_match2.group(1).strip()
        
        # 地積（面積）の抽出
        area_match = re.search(self.patterns['area'], text)
        if area_match:
            # 数値と単位を分ける
            structured_data['area'] = area_match.group(1).strip().replace(',', '')
            structured_data['area_unit'] = area_match.group(2).strip()
        else:
            # 別の表現パターンで再試行
            area_pattern2 = r'(?:地積|面積)\s*[:：]\s*([0-9０-９,.，．]+)\s*([平方メートル㎡])'
            area_match2 = re.search(area_pattern2, text)
            if area_match2:
                structured_data['area'] = area_match2.group(1).strip().replace(',', '')
                structured_data['area_unit'] = area_match2.group(2).strip()
        
        # 所有者（名義人）情報の抽出
        owner_name_match = re.search(self.patterns['owner_name'], text)
        if owner_name_match:
            structured_data['owner']['name'] = owner_name_match.group(1).strip()
        
        owner_address_match = re.search(self.patterns['owner_address'], text)
        if owner_address_match:
            structured_data['owner']['address'] = owner_address_match.group(1).strip()
        
        # 権利情報の抽出（甲区・乙区）
        # 甲区（所有権関連）の抽出
        rights_pattern = r'(?:甲区|権利者|所有権).*?\n(.*?)\n(?:乙区|債務者|順位|備考|$)'
        rights_match = re.search(rights_pattern, text, re.DOTALL)
        if rights_match:
            rights_text = rights_match.group(1)
            # 各権利の区切りで分割
            rights_entries = re.split(r'(?:\d+\s*番|\d+\s*順位)', rights_text)
            for entry in rights_entries:
                if entry.strip():
                    right = self._parse_right_entry(entry)
                    if right:
                        structured_data['rights'].append(right)
        
        # 乙区（抵当権等）の抽出
        mortgage_pattern = r'(?:乙区|抵当権).*?\n(.*?)(?:付記|$)'
        mortgage_match = re.search(mortgage_pattern, text, re.DOTALL)
        if mortgage_match:
            mortgage_text = mortgage_match.group(1)
            mortgage_entries = re.split(r'(?:\d+\s*番|\d+\s*順位)', mortgage_text)
            for entry in mortgage_entries:
                if entry.strip():
                    mortgage = self._parse_mortgage_entry(entry)
                    if mortgage:
                        structured_data['rights'].append(mortgage)
        
        return structured_data
    
    def _parse_right_entry(self, entry):
        """権利情報のエントリを解析する"""
        right = {
            'type': '所有権',
            'date': '',
            'cause': '',
            'owner': ''
        }
        
        # 登記日の抽出
        date_match = re.search(r'(?:登記日|受付日|日付)[：:]\s*([0-9０-９年月日]+)', entry)
        if date_match:
            right['date'] = date_match.group(1).strip()
        
        # 登記原因の抽出
        cause_match = re.search(r'(?:原因|登記原因)[：:]\s*(.+)', entry)
        if cause_match:
            right['cause'] = cause_match.group(1).strip()
        else:
            # 登記原因をキーワードから推測
            if '相続' in entry:
                right['cause'] = '相続'
            elif '売買' in entry:
                right['cause'] = '売買'
            elif '贈与' in entry:
                right['cause'] = '贈与'
        
        # 所有者の抽出
        owner_match = re.search(r'(?:所有者|権利者|名義人)[：:]\s*(.+)', entry)
        if owner_match:
            right['owner'] = owner_match.group(1).strip()
        
        return right if any(right.values()) else None
    
    def _parse_mortgage_entry(self, entry):
        """抵当権情報のエントリを解析する"""
        mortgage = {
            'type': '抵当権',
            'date': '',
            'amount': '',
            'creditor': '',
            'debtor': ''
        }
        
        # 登記日の抽出
        date_match = re.search(r'(?:登記日|受付日|日付)[：:]\s*([0-9０-９年月日]+)', entry)
        if date_match:
            mortgage['date'] = date_match.group(1).strip()
        
        # 債権額の抽出
        amount_match = re.search(r'(?:債権額|金額)[：:]\s*([0-9０-９,.，．]+)(?:円|金)', entry)
        if amount_match:
            mortgage['amount'] = amount_match.group(1).strip().replace(',', '')
        
        # 債権者の抽出
        creditor_match = re.search(r'(?:債権者|抵当権者)[：:]\s*(.+)', entry)
        if creditor_match:
            mortgage['creditor'] = creditor_match.group(1).strip()
        
        # 債務者の抽出
        debtor_match = re.search(r'(?:債務者|設定者)[：:]\s*(.+)', entry)
        if debtor_match:
            mortgage['debtor'] = debtor_match.group(1).strip()
        
        return mortgage if any(mortgage.values()) else None
    
    def process_image(self, image_path):
        """
        画像を処理してOCR結果を返す
        
        Args:
            image_path (str): 処理する画像のパス
            
        Returns:
            dict: OCR結果と構造化データを含む辞書
        """
        logger.info("Processing image: %s", image_path)
        
        # 前処理
        preprocessed = self.preprocess_image(image_path)
        
        # デバッグ用：前処理結果を一時ファイルとして保存
        debug_path = os.path.join(self.config['temp_dir'], 'debug_preprocessed.png')
        cv2.imwrite(debug_path, preprocessed)
        
        # テキスト抽出
        full_text = self.extract_text(preprocessed, self.config['lang'])
        
        # Tesseractが利用できない場合
        if not TESSERACT_AVAILABLE:
            # PDFが正常に変換されたことを示す最小限の結果を返す
            file_name = os.path.basename(image_path)
            processed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            result = {
                'file_name': file_name,
                'processed_at': processed_at,
                'file_path': image_path,
                'text': full_text,
                'confidence': 0,
                'structured_data': {
                    'property_number': 'OCR機能が利用できません',
                    'address': 'Tesseractがインストールされていません',
                    'lot_number': '',
                    'area': '',
                    'area_unit': '㎡',
                    'owner': {
                        'name': '',
                        'address': ''
                    },
                    'rights': []
                }
            }
            
            return result
        
        # 構造化データの抽出
        structured_data = self.extract_structured_data(full_text)
        
        # 信頼度の計算
        confidence = self._calculate_confidence(full_text)
        
        # データの検証と強化
        structured_data = self._validate_and_enhance_data(structured_data)
        
        # 結果の組み立て
        file_name = os.path.basename(image_path)
        processed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'file_name': file_name,
            'processed_at': processed_at,
            'file_path': image_path,
            'text': full_text,
            'confidence': confidence,
            'structured_data': structured_data
        }
        
        logger.info("OCR process completed with confidence: %f", confidence)
        
        return result
    
    def _calculate_confidence(self, text):
        """
        テキストの信頼度を計算する
        
        Args:
            text (str): OCRで抽出されたテキスト
            
        Returns:
            float: 信頼度スコア（0-100）
        """
        # テキストが空の場合は0を返す
        if not text or len(text) < 10:
            return 0
        
        # 文字数と行数
        total_chars = len(text)
        total_lines = text.count('\n') + 1
        
        # 疑わしい文字のパターン（記号の混合や不明瞭な文字列）
        suspicious_patterns = [
            r'[^\w\s.,、。：;\-()（）「」]',  # 一般的でない記号
            r'[a-zA-Z]{1}[a-zA-Z]{1}',  # 英字の混入
            r'[0-9]{5,}',  # 不自然に長い数字列
        ]
        
        suspicious_chars = 0
        for pattern in suspicious_patterns:
            suspicious_chars += len(re.findall(pattern, text))
        
        # 構造的な信頼度（項目の検出率）
        # 新しいデータ構造に合わせて修正
        structure_items = ['property_number', 'address', 'lot_number', 'area', 'owner']
        structure_score = 0
        
        # 一時的に構造データを抽出して評価
        data = self.extract_structured_data(text)
        for item in structure_items:
            # 安全にアクセスするように修正
            if item in data:
                if item == 'owner':
                    # ownerは辞書なので、nameフィールドを確認
                    if data[item] and 'name' in data[item] and data[item]['name']:
                        structure_score += 20
                elif data[item]:  # その他の項目
                    structure_score += 20  # 各項目20%として計算
        
        # 文字認識の信頼度（疑わしい文字が少ないほど高い）
        char_confidence = 100 - (suspicious_chars / max(total_chars, 1) * 100)
        
        # 総合的な信頼度
        total_confidence = (char_confidence * 0.7) + (structure_score * 0.3)
        
        # 範囲を0-100に制限
        return max(0, min(100, total_confidence))
    
    def save_result(self, result, output_format='json'):
        """
        OCR処理結果を保存する
        
        Args:
            result (dict): OCR処理結果
            output_format (str, optional): 出力形式。デフォルトは'json'。
            
        Returns:
            str: 保存されたファイルのパス
        """
        filename = os.path.splitext(result['file_name'])[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{filename}_{timestamp}"
        
        if output_format == 'json':
            output_path = os.path.join(self.config['output_dir'], f"{output_filename}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info("Result saved to JSON: %s", output_path)
            
        elif output_format == 'csv':
            output_path = os.path.join(self.config['output_dir'], f"{output_filename}.csv")
            
            # 構造化データの取得
            structured_data = result.get('structured_data', {})
            
            # 新しいデータ構造に対応
            owner_data = structured_data.get('owner', {})
            owner_name = owner_data.get('name', '') if isinstance(owner_data, dict) else ''
            owner_address = owner_data.get('address', '') if isinstance(owner_data, dict) else ''
            
            # 権利情報の処理
            rights_info = []
            for right in structured_data.get('rights', []):
                if isinstance(right, dict):
                    right_type = right.get('type', '')
                    right_date = right.get('date', '')
                    right_cause = right.get('cause', '')
                    right_owner = right.get('owner', '')
                    rights_info.append(f"{right_type}({right_date}, {right_cause}, {right_owner})")
            
            # データフレームに変換
            df_data = {
                '不動産番号': [structured_data.get('property_number', '')],
                '所在': [structured_data.get('address', '')],
                '地番': [structured_data.get('lot_number', '')],
                '地積': [f"{structured_data.get('area', '')} {structured_data.get('area_unit', '㎡')}"],
                '所有者名': [owner_name],
                '所有者住所': [owner_address],
                '権利情報': ['; '.join(rights_info)]
            }
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info("Result saved to CSV: %s", output_path)
            
        else:
            raise ValueError(f"未対応の出力形式です: {output_format}")
        
        return output_path
    
    def batch_process(self, image_dir, output_format='json'):
        """
        ディレクトリ内の画像を一括処理する
        
        Args:
            image_dir (str): 処理する画像が含まれるディレクトリのパス
            output_format (str, optional): 出力形式。デフォルトは'json'。
            
        Returns:
            list: 処理結果のリスト
        """
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pdf']
        results = []
        
        logger.info("Starting batch processing in directory: %s", image_dir)
        
        # ディレクトリ内のファイルを取得
        files = [f for f in os.listdir(image_dir) 
                if os.path.isfile(os.path.join(image_dir, f)) and 
                os.path.splitext(f)[1].lower() in allowed_extensions]
        
        logger.info("Found %d files to process", len(files))
        
        for file in files:
            file_path = os.path.join(image_dir, file)
            try:
                logger.info("Processing file: %s", file_path)
                result = self.process_image(file_path)
                output_path = self.save_result(result, output_format)
                
                results.append({
                    'file_name': file,
                    'output_path': output_path,
                    'confidence': result['confidence']
                })
                
                logger.info("Successfully processed file: %s", file_path)
                
            except Exception as e:
                logger.error("Error processing file %s: %s", file_path, str(e), exc_info=True)
                results.append({
                    'file_name': file,
                    'error': str(e)
                })
        
        logger.info("Batch processing completed. Processed %d/%d files successfully", 
                    len([r for r in results if 'error' not in r]), len(files))
        
        return results


# 使用例
if __name__ == "__main__":
    # コマンドライン引数の処理
    if len(sys.argv) < 2:
        print("使用方法: python ocr_implementation.py <画像ファイルまたはディレクトリ>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # OCRエンジンの初期化
    ocr = RegistryOCR()
    
    # ファイルまたはディレクトリの処理
    if os.path.isfile(input_path):
        # 単一ファイルの処理
        result = ocr.process_image(input_path)
        output_path = ocr.save_result(result)
        print(f"処理完了: {input_path} -> {output_path}")
    
    elif os.path.isdir(input_path):
        # ディレクトリのバッチ処理
        results = ocr.batch_process(input_path)
        
        # 結果の要約
        success_count = sum(1 for r in results if 'error' not in r)
        print(f"\n処理完了: {success_count}/{len(results)} ファイルが正常に処理されました")
        
        # エラーがあれば表示
        errors = [r for r in results if 'error' in r]
        if errors:
            print("\nエラーが発生したファイル:")
            for error in errors:
                print(f"- {error['file_name']}: {error['error']}")
    
    else:
        print(f"エラー: 指定されたパスが存在しません: {input_path}")
        sys.exit(1)
