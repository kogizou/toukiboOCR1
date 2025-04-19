"""
登記簿OCRアプリ - OpenAI OCR実装

このスクリプトはOpenAI GPT-4 Visionを使用して画像からテキストを抽出し、
登記簿データを構造化する機能を提供します。
"""

import os
import sys
import base64
import json
import logging
import time
import tempfile
import cv2
import numpy as np
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF
from openai import OpenAI

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("openai_ocr_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('OpenAI_OCR')

class OpenAIOCR:
    """OpenAI Vision APIを使用したOCR処理クラス"""
    
    def __init__(self, config=None):
        """
        初期化メソッド
        
        Args:
            config (dict, optional): 設定情報を含む辞書。デフォルトはNone。
        """
        # デフォルト設定
        self.default_config = {
            'api_key': os.environ.get('OPENAI_API_KEY', ''),  # OpenAI APIキー
            'model': 'gpt-4-vision-preview',  # 使用するモデル
            'temperature': 0.3,  # 生成の多様性
            'max_tokens': 4000,  # 最大トークン数
            'detail_level': 'high',  # 画像の詳細レベル
            'temp_dir': './temp',  # 一時ファイル保存ディレクトリ
            'output_dir': './output',  # 出力ディレクトリ
            'dpi': 300,  # DPI設定
        }
        
        # 設定の初期化
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # OpenAIクライアントの初期化
        try:
            self.client = OpenAI(api_key=self.config['api_key'])
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
        
        # 一時ディレクトリと出力ディレクトリの作成
        os.makedirs(self.config['temp_dir'], exist_ok=True)
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def preprocess_image(self, image_path):
        """
        画像の前処理を行う
        
        Args:
            image_path (str): 処理する画像のパス
            
        Returns:
            str: 前処理された画像のパス
        """
        logger.info("Processing image: %s", image_path)
        
        # PDFファイルの処理
        if image_path.lower().endswith('.pdf'):
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
                
                # 使用後にPDFを閉じる
                pdf_document.close()
                
                return temp_image_path
            except Exception as e:
                logger.error("PDF変換中にエラーが発生しました: %s", str(e), exc_info=True)
                raise ValueError(f"PDF変換中にエラーが発生しました: {str(e)}")
        else:
            # 画像の最適化のため、一時ファイルにコピー
            try:
                # 画像を読み込み
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"画像を読み込めませんでした: {image_path}")
                
                # 画像の最適化
                # 解像度調整
                image_height, image_width = image.shape[:2]
                current_dpi = max(image_width, image_height) / 8.5  # A4用紙を想定
                scale_factor = self.config['dpi'] / current_dpi if current_dpi > 0 else 1
                
                if scale_factor != 1:
                    logger.info("Resizing image with scale factor: %f", scale_factor)
                    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                
                # コントラスト強調
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                
                # 一時ファイルに保存
                temp_dir = tempfile.mkdtemp(dir=self.config['temp_dir'])
                temp_image_path = os.path.join(temp_dir, "enhanced_image.png")
                cv2.imwrite(temp_image_path, enhanced_image)
                logger.info("Image enhanced and saved to: %s", temp_image_path)
                
                return temp_image_path
            except Exception as e:
                logger.error("画像処理中にエラーが発生しました: %s", str(e), exc_info=True)
                # 処理に失敗した場合は元の画像パスを返す
                return image_path
    
    def encode_image(self, image_path):
        """
        画像をBase64エンコードする
        
        Args:
            image_path (str): 画像のパス
            
        Returns:
            str: Base64エンコードされた画像データ
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_text_from_image(self, image_path):
        """
        OpenAI Vision APIを使用して画像からテキストを抽出する
        
        Args:
            image_path (str): 画像のパス
            
        Returns:
            str: 抽出されたテキスト
        """
        if not self.client:
            raise ValueError("OpenAI clientが初期化されていません。API Keyを確認してください。")
        
        try:
            # 画像をBase64エンコード
            base64_image = self.encode_image(image_path)
            
            # プロンプトの作成
            prompt = """
            このスキャンされた登記簿謄本から全てのテキストを抽出してください。
            登記簿には以下の情報が含まれている可能性があります：
            - 不動産番号
            - 所在地（住所）
            - 地番
            - 地積（面積）
            - 所有者情報（名義人、住所）
            - 権利情報（所有権、抵当権など）
            
            できるだけ正確に、表形式やレイアウトも保持してテキストを抽出してください。
            日本語の登記簿特有の用語や形式を保持してください。
            """
            
            # APIリクエスト
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=[
                    {"role": "system", "content": "あなたは登記簿OCRの専門家です。画像から正確にテキストを抽出し、構造化された形式で提供します。"},
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", 
                         "image_url": {
                             "url": f"data:image/png;base64,{base64_image}",
                             "detail": self.config['detail_level']
                         }}
                    ]}
                ],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature']
            )
            
            # レスポンスからテキストを取得
            extracted_text = response.choices[0].message.content
            logger.info("Text extracted successfully with %d characters", len(extracted_text))
            
            return extracted_text
        except Exception as e:
            logger.error("Text extraction failed: %s", str(e), exc_info=True)
            raise ValueError(f"テキスト抽出中にエラーが発生しました: {str(e)}")
    
    def extract_structured_data(self, text):
        """
        テキストから構造化データを抽出する
        
        Args:
            text (str): 抽出されたテキスト
            
        Returns:
            dict: 構造化データ
        """
        if not self.client:
            raise ValueError("OpenAI clientが初期化されていません。API Keyを確認してください。")
        
        try:
            # プロンプトの作成
            prompt = f"""
            以下の登記簿テキストから構造化データを抽出し、JSONフォーマットで返してください。

            抽出するデータ:
            1. property_number: 不動産番号（例: "1234-5678-9012"）
            2. address: 所在地（例: "東京都千代田区丸の内1-1"）
            3. lot_number: 地番（例: "1番2"）
            4. area: 地積/面積（例: "100.25"）
            5. area_unit: 面積の単位（例: "㎡"）
            6. owner: 所有者情報
               - name: 名義人/所有者名（例: "山田太郎"）
               - address: 所有者の住所（例: "東京都新宿区新宿1-1"）
            7. rights: 権利情報の配列
               - type: 権利の種類（例: "所有権"、"抵当権"など）
               - date: 登記日
               - cause: 登記原因（例: "売買"、"相続"など）
               - owner: 権利者名
               - 抵当権の場合:
                 - amount: 債権額
                 - creditor: 債権者
                 - debtor: 債務者

            返す形式はJSON形式で、不明な項目は空文字にしてください。
            
            テキスト:
            {text}
            """
            
            # APIリクエスト
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",  # 高速処理のためターボモデルを使用
                messages=[
                    {"role": "system", "content": "あなたは登記簿データの構造化の専門家です。テキストからJSON形式で構造化データを抽出します。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.2,  # 正確さを重視
                response_format={"type": "json_object"}  # JSON形式の出力を強制
            )
            
            # レスポンスからJSONを取得
            structured_data_json = response.choices[0].message.content
            structured_data = json.loads(structured_data_json)
            logger.info("Structured data extracted successfully")
            
            return structured_data
        except Exception as e:
            logger.error("Structured data extraction failed: %s", str(e), exc_info=True)
            # エラー時は最小限の構造化データを返す
            return {
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
        preprocessed_image_path = self.preprocess_image(image_path)
        
        # テキスト抽出
        full_text = self.extract_text_from_image(preprocessed_image_path)
        
        # 構造化データの抽出
        structured_data = self.extract_structured_data(full_text)
        
        # 信頼度の計算 (OpenAIは信頼度を返さないので、固定値)
        confidence = 90.0
        
        # 結果の組み立て
        file_name = os.path.basename(image_path)
        processed_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        result = {
            'file_name': file_name,
            'processed_at': processed_at,
            'file_path': image_path,
            'text': full_text,
            'confidence': confidence,
            'structured_data': structured_data,
            'processed_by': 'OpenAI GPT-4 Vision'
        }
        
        logger.info("OCR process completed")
        
        return result
    
    def save_result(self, result, output_format='json'):
        """
        処理結果を保存する
        
        Args:
            result (dict): 処理結果
            output_format (str, optional): 出力形式。デフォルトは'json'。
            
        Returns:
            str: 保存されたファイルのパス
        """
        # ファイル名の生成
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        base_name = os.path.splitext(result['file_name'])[0]
        output_filename = f"{base_name}_{timestamp}.{output_format}"
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        try:
            # JSONとして保存
            if output_format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                    
            # その他の形式のサポート（必要に応じて追加）
            else:
                raise ValueError(f"未サポートの出力形式: {output_format}")
                
            logger.info("Result saved to: %s", output_path)
            return output_path
        except Exception as e:
            logger.error("Failed to save result: %s", str(e), exc_info=True)
            raise
            
    @staticmethod
    def check_api_key_validity(api_key):
        """
        APIキーの有効性を確認する
        
        Args:
            api_key (str): 検証するAPIキー
            
        Returns:
            bool: キーが有効な場合はTrue
        """
        try:
            client = OpenAI(api_key=api_key)
            # APIキーが有効かどうかを簡単なリクエストで確認
            response = client.models.list()
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {str(e)}")
            return False

# テスト用コード
if __name__ == "__main__":
    # コマンドライン引数からAPIキーを取得
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    else:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("OpenAI APIキーが必要です。環境変数 'OPENAI_API_KEY' を設定するか、コマンドライン引数として渡してください。")
            sys.exit(1)
    
    # OCRエンジンの初期化
    config = {
        'api_key': api_key,
        'temp_dir': './temp',
        'output_dir': './output'
    }
    ocr = OpenAIOCR(config)
    
    # テスト画像のパス
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        
        # OCR処理の実行
        try:
            result = ocr.process_image(image_path)
            output_path = ocr.save_result(result)
            print(f"OCR処理が完了しました。結果: {output_path}")
            print(f"抽出されたテキスト（一部）: {result['text'][:200]}...")
            print(f"構造化データ: {json.dumps(result['structured_data'], ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"OCR処理中にエラーが発生しました: {str(e)}")
    else:
        print("処理する画像のパスを指定してください。") 