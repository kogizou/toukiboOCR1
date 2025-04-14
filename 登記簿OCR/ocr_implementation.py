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
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = self.config['tesseract_cmd']
        
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
        
        # 画像の読み込み
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
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 傾き補正
        try:
            coords = np.column_stack(np.where(gray > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            if abs(angle) > 0.5:  # 0.5度以上の傾きがある場合のみ補正
                logger.info("Correcting image skew: %f degrees", angle)
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            logger.warning("Failed to correct image skew: %s", str(e))
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # ノイズ除去（バイラテラルフィルタ - エッジを保存しながらノイズを除去）
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 二値化（大津の二値化）
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # モルフォロジー演算でノイズ除去
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 結果を一時ファイルとして保存（デバッグ用）
        debug_path = os.path.join(self.config['temp_dir'], 'debug_preprocess.png')
        cv2.imwrite(debug_path, binary)
        logger.info("Preprocessed image saved to: %s", debug_path)
        
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
        # PILイメージに変換
        pil_image = Image.fromarray(image)
        
        # テキスト抽出（詳細な設定）
        config = f'--psm {self.config["psm"]} --oem {self.config["oem"]} -c preserve_interword_spaces=1'
        
        # テキスト抽出
        text = pytesseract.image_to_string(
            pil_image, 
            lang=lang,
            config=config
        )
        
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
        
        return text
    
    def extract_structured_data(self, text):
        """
        抽出されたテキストから構造化データを生成する
        
        Args:
            text (str): OCRで抽出されたテキスト
            
        Returns:
            dict: 構造化されたデータ
        """
        data = {
            'property_number': None,
            'address': None,
            'lot_number': None,
            'area': None,
            'owner_name': None,
            'owner_address': None,
            'rights': [],
            'raw_text': text
        }
        
        # 各項目を正規表現で抽出
        for key, pattern in self.patterns.items():
            match = re.search(pattern, text, re.MULTILINE)
            if match:
                if key == 'rights':
                    # 権利情報は複数ある可能性があるため、リストとして保存
                    rights_matches = re.findall(pattern, text, re.MULTILINE)
                    data[key] = [rm.strip() for rm in rights_matches if rm.strip()]
                elif key == 'area':
                    # 面積のフォーマット統一
                    area_value = match.group(1).strip().replace(',', '')
                    unit = match.group(2)
                    try:
                        area_value = float(area_value)
                        data[key] = f"{area_value} {unit}"
                    except ValueError:
                        data[key] = f"{match.group(1)} {unit}"
                else:
                    data[key] = match.group(1).strip()
                    
                logger.info("Extracted %s: %s", key, data[key])
        
        # データの補完と整合性チェック
        self._validate_and_enhance_data(data)
        
        return data
    
    def _validate_and_enhance_data(self, data):
        """
        抽出されたデータの検証と補完を行う
        
        Args:
            data (dict): 構造化されたデータ
            
        Returns:
            None: データは直接更新される
        """
        # 不動産番号のフォーマット検証
        if data['property_number']:
            if not re.match(r'\d{4}-\d{4}-\d{4}', data['property_number']):
                # ハイフンの統一
                data['property_number'] = re.sub(r'(\d{4})[-－]?(\d{4})[-－]?(\d{4})', r'\1-\2-\3', data['property_number'])
                logger.info("Reformatted property number: %s", data['property_number'])
        
        # 権利情報が空の場合、テキストから抽出を試みる
        if not data['rights']:
            rights_sections = re.findall(r'権\s*利\s*者\s*(.+?)(?=\n\n|\Z)', data['raw_text'], re.DOTALL)
            if rights_sections:
                data['rights'] = [r.strip() for r in rights_sections if r.strip()]
                logger.info("Extracted rights from full text: %s", data['rights'])
        
        # 面積の数値検証
        if data['area']:
            area_match = re.search(r'([\d.,]+)\s*([平方メートル㎡])', data['area'])
            if area_match:
                try:
                    area_value = float(area_match.group(1).replace(',', ''))
                    # 極端に大きい/小さい値はエラーの可能性
                    if area_value > 10000000 or area_value < 0.1:
                        logger.warning("Suspicious area value: %f", area_value)
                except ValueError:
                    logger.warning("Invalid area format: %s", data['area'])
    
    def process_image(self, image_path):
        """
        画像を処理してOCR結果を返す
        
        Args:
            image_path (str): 処理する画像のパス
            
        Returns:
            dict: OCR処理結果
        """
        # 画像の前処理
        preprocessed = self.preprocess_image(image_path)
        
        # テーブル構造の検出
        table_cells = self.detect_table_structure(preprocessed)
        
        # 全体テキストの抽出
        full_text = self.extract_text(preprocessed, self.config['lang'])
        
        # 構造化データの抽出
        structured_data = self.extract_structured_data(full_text)
        
        # テーブルセルごとのテキスト抽出（詳細分析用）
        cell_texts = []
        for x, y, w, h in table_cells:
            cell_image = preprocessed[y:y+h, x:x+w]
            cell_text = self.extract_text(cell_image, self.config['lang'])
            if cell_text.strip():  # 空のテキストは除外
                cell_texts.append({
                    'position': (x, y, w, h),
                    'text': cell_text.strip()
                })
        
        # 結果の作成
        result = {
            'file_name': os.path.basename(image_path),
            'processed_at': datetime.now().isoformat(),
            'structured_data': structured_data,
            'table_cells': cell_texts,
            'confidence': self._calculate_confidence(full_text)
        }
        
        return result
    
    def _calculate_confidence(self, text):
        """
        OCR結果の信頼度を計算する
        
        Args:
            text (str): OCRで抽出されたテキスト
            
        Returns:
            float: 信頼度（0-100）
        """
        # 基本的な文字認識率を見積もる
        if not text.strip():
            return 0
        
        # テキストの文字数
        total_chars = len(text.strip())
        
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
        structure_items = ['property_number', 'address', 'lot_number', 'area', 'owner_name']
        structure_score = 0
        
        # 一時的に構造データを抽出して評価
        data = self.extract_structured_data(text)
        for item in structure_items:
            if data[item]:
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
            
            # 構造化データをデータフレームに変換
            df_data = {
                '不動産番号': [result['structured_data'].get('property_number', '')],
                '所在': [result['structured_data'].get('address', '')],
                '地番': [result['structured_data'].get('lot_number', '')],
                '地積': [result['structured_data'].get('area', '')],
                '所有者名': [result['structured_data'].get('owner_name', '')],
                '所有者住所': [result['structured_data'].get('owner_address', '')],
                '権利情報': [', '.join(result['structured_data'].get('rights', []))]
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
