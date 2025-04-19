# -*- coding: utf-8 -*-
import re
import os
import json
import logging
import numpy as np
import cv2
import pytesseract
from datetime import datetime
from PIL import Image
import time
import hashlib
import sys
import tempfile
import subprocess
import shutil

# ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class RegistryOCR:
    """登記簿OCR処理のメインクラス"""
    
    def __init__(self, config=None):
        """
        登記簿OCRエンジンの初期化
        
        Args:
            config (dict, optional): OCRの設定。指定がない場合はデフォルト設定を使用。
        """
        self.config = {
            'lang': 'jpn',  # 日本語
            'dpi': 300,     # OCRのDPI設定
            'psm': 6,       # ページセグメンテーションモード（PSM）
            'oem': 3,       # OCRエンジンモード（OEM）
            'config': '--tessdata-dir "tessdata"'  # Tesseractの設定
        }
        
        if config:
            self.config.update(config)
        
        # Tesseractのパスを設定
        if sys.platform.startswith('win'):
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # 一時ディレクトリの作成
        self.temp_dir = self.config['temp_dir']
        self.output_dir = self.config['output_dir']
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Tesseractのデータディレクトリ
        self.tesseract_data_path = '/usr/local/share/tessdata'
        if not os.path.exists(self.tesseract_data_path):
            self.tesseract_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata')
            os.makedirs(self.tesseract_data_path, exist_ok=True)
        
        logger.info("RegistryOCR initialized with config: %s", json.dumps(self.config, ensure_ascii=False))
        
        # 必要なライブラリの存在確認
        self._check_dependencies()
        
        # ログ設定
        self.logger = logging.getLogger('registry_ocr')
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _optimize_tesseract_config(self, document_type=None):
        """
        ドキュメントタイプに基づいてTesseractの設定文字列を生成する
        
        Args:
            document_type (str, optional): テキストの向き ('vertical' または 'horizontal')
            
        Returns:
            str: Tesseractの設定文字列
        """
        # 基本設定
        lang = self.config.get('lang', 'jpn')
        psm = self.config.get('psm', 6)
        oem = self.config.get('oem', 3)
        
        # テキストの向きに応じて設定を調整
        if document_type == "vertical":
            # 縦書きテキスト用の設定
            lang = f"{lang}+jpn_vert"
            psm = 5  # 単一の縦書きテキストブロック
            config = f'--psm {psm} --oem {oem} -l {lang}'
            config += ' --tessdata-dir "/usr/local/share/tessdata"'
            config += ' -c preserve_interword_spaces=1 -c textord_tabfind_vertical_text=1'
            config += ' -c textord_tabfind_force_vertical_text=1'
        elif document_type == "horizontal":
            # 横書きテキスト用の設定
            psm = 6  # 単一の横書きテキストブロック
            config = f'--psm {psm} --oem {oem} -l {lang}'
            config += ' --tessdata-dir "/usr/local/share/tessdata"'
            config += ' -c preserve_interword_spaces=1'
        else:
            # 自動検出モード
            psm = 3  # 自動ページセグメンテーション
            config = f'--psm {psm} --oem {oem} -l {lang}'
            config += ' --tessdata-dir "/usr/local/share/tessdata"'
            config += ' -c preserve_interword_spaces=1 -c textord_tabfind_vertical_text=1'
        
        # 日本語に最適化した文字セット（許可する文字）
        config += ' -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴッ々〆ー・「」。、（）～￥[]{}:：""\'\'＜＞〒♯＆@＠/／|｜一二三四五六七八九十百千万億兆京年月日時分秒'
        
        # 文字認識の精度向上のための追加設定
        config += ' -c language_model_penalty_non_dict_word=0.8'
        config += ' -c language_model_penalty_non_freq_dict_word=0.6'
        
        return config

    def _detect_document_type(self, text):
        """
        OCRで認識されたテキストから文書タイプを自動検出する
        
        Args:
            text (str): OCRで認識されたテキスト
            
        Returns:
            str: 検出された文書タイプ（'registry', 'certificate', 'mixed', 'vertical', 'other'）
        """
        if not text or len(text) < 20:
            return 'other'  # テキストが短すぎる場合は判断できない
            
        # 登記簿関連のキーワード
        registry_keywords = [
            '登記簿', '謄本', '登記事項', '甲区', '乙区', '土地', '建物', 
            '所有権', '権利者', '順位', '登記原因', '共同担保', '抵当権', 
            '根抵当権', '債権額', '持分', '地番', '不動産番号'
        ]
        
        # 権利証関連のキーワード
        certificate_keywords = [
            '権利証', '所有権', '登記済証', '登記完了証', '証明書',
            '登記官', '印鑑証明', '印鑑登録'
        ]
        
        # 縦書きのパターンを検出（縦に並んだ句読点などを検出）
        vertical_patterns = [
            r'[\u3001\u3002\uff0c\uff0e]{2,}',  # 複数の句読点が縦に並ぶパターン
            r'([\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff])\s+\1',  # 同じ文字が縦に並ぶパターン
        ]
        
        # スコアリングによる文書タイプ判定
        registry_score = sum(1 for kw in registry_keywords if kw in text)
        certificate_score = sum(1 for kw in certificate_keywords if kw in text)
        vertical_score = 0
        
        # 縦書きパターンの検出
        import re
        for pattern in vertical_patterns:
            if re.search(pattern, text):
                vertical_score += 1
                
        # 甲区/乙区のパターンを検出（登記簿の特徴的な構造）
        if re.search(r'甲区.+乙区', text, re.DOTALL) or re.search(r'所有権.+担保', text, re.DOTALL):
            registry_score += 3
        
        # 表のパターンを検出
        if re.search(r'[-─|｜]{3,}', text):
            registry_score += 1
            
        # 日付のパターンを検出（登記関連文書の特徴）
        date_patterns = [
            r'(平成|令和|昭和)(\d{1,2})年(\d{1,2})月(\d{1,2})日',
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{4}/\d{1,2}/\d{1,2}'
        ]
        for pattern in date_patterns:
            if re.search(pattern, text):
                registry_score += 1
                certificate_score += 1
        
        # 所有権/抵当権の詳細パターンを検出
        if re.search(r'所有権.*登記原因.*', text, re.DOTALL):
            registry_score += 2
            
        if re.search(r'抵当権.*債権額.*', text, re.DOTALL):
            registry_score += 2
            
        self.logger.debug(f"文書タイプスコア - 登記簿: {registry_score}, 権利証: {certificate_score}, 縦書き: {vertical_score}")
            
        # スコアに基づいて文書タイプを判定
        if registry_score >= 3:
            if vertical_score >= 1:
                return 'registry'  # 縦書きの登記簿
            else:
                return 'registry'  # 横書きの登記簿（まれ）
        elif certificate_score >= 2:
            return 'certificate'  # 権利証
        elif vertical_score >= 1:
            return 'vertical'  # 縦書き文書（登記簿や権利証の特徴が少ない）
        elif registry_score >= 1 or certificate_score >= 1:
            return 'mixed'  # 登記関連だが種類が明確でない
        else:
            return 'other'  # その他の文書

    def _validate_and_enhance_data(self, data):
        """
        OCRで抽出されたデータを検証し、適切な形式に強化する
        
        Args:
            data (dict): 検証および強化する構造化データ
            
        Returns:
            dict: 検証および強化された構造化データ
        """
        logger.info("データの検証と強化を開始します")
        
        if not data:
            return {}
            
        # 空の値をNoneに変換
        data = self._clean_empty_values(data)
        
        # 物件番号の検証と修正
        if 'property_number' in data and data['property_number']:
            # 物件番号のフォーマットをチェック（例：YYYY-XXXX-XXXX）
            pattern = r'^\d{4}-\d{4}-\d{4}$'
            if not re.match(pattern, data['property_number']):
                # フォーマットが正しくない場合、修正を試みる
                # 数字のみを抽出
                digits = re.findall(r'\d', data['property_number'])
                if len(digits) >= 12:
                    # 12桁以上ある場合、フォーマットを修正
                    formatted = ''.join(digits[:4]) + '-' + ''.join(digits[4:8]) + '-' + ''.join(digits[8:12])
                    logger.info(f"物件番号のフォーマットを修正しました: {data['property_number']} -> {formatted}")
                    data['property_number'] = formatted
                else:
                    logger.warning(f"物件番号のフォーマットが不正で修正できません: {data['property_number']}")
        
        # 面積の数値への変換と丸め
        if 'area' in data and data['area']:
            try:
                if isinstance(data['area'], str):
                    # カンマを削除し、数値に変換
                    area_str = data['area'].replace(',', '')
                    # 数値部分を抽出（単位などを除去）
                    number_match = re.search(r'(\d+\.?\d*)', area_str)
                    if number_match:
                        area_value = float(number_match.group(1))
                        # 小数点以下2桁に丸める
                        data['area'] = round(area_value, 2)
                elif isinstance(data['area'], (int, float)):
                    data['area'] = round(float(data['area']), 2)
            except (ValueError, TypeError) as e:
                logger.warning(f"面積の変換中にエラーが発生しました: {str(e)}")
                data['area'] = None
        
        # 権利情報の強化
        if 'rights' in data and data['rights']:
            for i, right in enumerate(data['rights']):
                # 日付のフォーマット統一
                if 'date' in right and right['date']:
                    try:
                        # 日付形式の検証と変換
                        date_str = right['date']
                        # 和暦から西暦への変換などの処理も必要に応じて追加
                    except Exception as e:
                        logger.warning(f"権利情報の日付変換中にエラーが発生しました: {str(e)}")
                
                # 金額の数値変換
                if 'amount' in right and right['amount']:
                    try:
                        if isinstance(right['amount'], str):
                            # カンマや通貨記号を削除
                            amount_str = right['amount'].replace(',', '').replace('¥', '').replace('円', '')
                            # 数値部分を抽出
                            number_match = re.search(r'(\d+)', amount_str)
                            if number_match:
                                right['amount'] = int(number_match.group(1))
                    except (ValueError, TypeError) as e:
                        logger.warning(f"金額の変換中にエラーが発生しました: {str(e)}")
        
        logger.info("データの検証と強化が完了しました")
        return data

    def _clean_empty_values(self, data_dict):
        """辞書内の空の値をNoneに変換する（再帰的）
        
        Args:
            data_dict (dict): 処理する辞書
            
        Returns:
            dict: 処理後の辞書
        """
        if not isinstance(data_dict, dict):
            return data_dict
            
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                result[key] = self._clean_empty_values(value)
            elif isinstance(value, list):
                if not value:
                    result[key] = None
                else:
                    # リスト内の辞書も処理
                    new_list = []
                    for item in value:
                        if isinstance(item, dict):
                            new_list.append(self._clean_empty_values(item))
                        else:
                            new_list.append(item)
                    result[key] = new_list
            elif isinstance(value, str) and value.strip() == '':
                result[key] = None
            else:
                result[key] = value
                
        return result

    def process_image(self, image_path):
        """
        画像ファイルを処理してテキストを抽出し、構造化データを返す
        
        Args:
            image_path (str): 処理する画像のパス
            
        Returns:
            tuple: (構造化データ辞書, 信頼度スコア, 抽出テキスト)
        """
        self.logger.info(f"画像処理開始: {image_path}")
        start_time = time.time()
        
        try:
            # 画像が存在するか確認
            if not os.path.exists(image_path):
                self.logger.error(f"画像ファイルが見つかりません: {image_path}")
                return {}, 0.0, ""
            
            # 1. 画像の前処理強化（新機能）
            if self.config.get('enhance_preprocessing', True):
                try:
                    enhanced_image_path = self.enhance_image(image_path)
                    self.logger.info(f"強化前処理を適用しました: {enhanced_image_path}")
                    image_path = enhanced_image_path
                except Exception as e:
                    self.logger.warning(f"画像強化処理に失敗しました。標準処理を続行します: {str(e)}")
            
            # 2. 画像の読み込み
            try:
                image = cv2.imread(image_path)
                if image is None:
                    # PILで試みる
                    image = np.array(Image.open(image_path))
                    # BGR形式に変換（OpenCV標準）
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        image = image[:, :, ::-1]
            except Exception as e:
                self.logger.error(f"画像読み込みエラー: {str(e)}")
                return {}, 0.0, ""
                
            # 3. 文書の向き検出
            orientation = "vertical" if self._is_vertical_text(image) and self.config.get('detect_orientation', True) else "horizontal"
            self.logger.info(f"検出されたテキスト方向: {orientation}")
            
            # 4. OCR設定の最適化
            custom_config = self._optimize_tesseract_config(
                "vertical" if orientation == "vertical" else "horizontal"
            )
            
            # 日本語最適化設定の適用
            if self.config.get('optimize_japanese', True):
                custom_config += ' -c textord_tablefind_recognize_tables=0'
                custom_config += ' -c textord_tabfind_vertical_text=1' if orientation == "vertical" else ''
                custom_config += ' -c language_model_ngram_on=1'
                custom_config += ' -c language_model_ngram_space_delimited_language=0'
                
            # Tesseract OCRの設定をセット
            ocr_config = {
                'lang': self.config.get('lang', 'jpn'),
                'config': custom_config,
                'dpi': self.config.get('dpi', 300)
            }
            
            # 5. OCRテキスト抽出
            self.logger.info(f"OCR実行中... 設定: {custom_config}")
            
            # 5.1 画像をメモリに展開
            is_success, im_buf_arr = cv2.imencode(".png", image)
            byte_im = im_buf_arr.tobytes()
            
            # 5.2 PyTesseractでOCR実行
            from io import BytesIO
            ocr_text = pytesseract.image_to_string(
                Image.open(BytesIO(byte_im)),
                lang=ocr_config['lang'],
                config=ocr_config['config']
            )
            
            # OCRテキストが空の場合はエラー
            if not ocr_text or ocr_text.strip() == "":
                self.logger.warning("OCRテキストが空です。別の前処理方法を試みます。")
                # 別の前処理方法を試す（PSMを変更して再試行）
                alt_config = custom_config.replace('--psm 6', '--psm 3').replace('--psm 5', '--psm 4')
                ocr_text = pytesseract.image_to_string(
                    Image.open(BytesIO(byte_im)),
                    lang=ocr_config['lang'],
                    config=alt_config
                )
                
                if not ocr_text or ocr_text.strip() == "":
                    self.logger.error("OCRテキスト抽出失敗")
                    return {}, 0.0, ""
            
            # 6. 文書タイプの検出
            document_type = self._detect_document_type(ocr_text)
            self.logger.info(f"検出された文書タイプ: {document_type}")
            
            # 7. テキストの後処理
            processed_text = self._postprocess_text(ocr_text, document_type)
            
            # 8. 構造化データの抽出
            structured_data = self.extract_structured_data(processed_text)
            
            # 9. 信頼度の計算
            confidence = self._calculate_confidence(structured_data)
            
            # 10. データの検証と拡張
            validated_data = self._validate_and_enhance_data(structured_data)
            
            # 処理時間の記録
            processing_time = time.time() - start_time
            self.logger.info(f"画像処理完了: {processing_time:.2f}秒, 信頼度: {confidence:.2f}")
            
            # 結果をキャッシュに保存（オプション）
            try:
            self._save_to_cache(image_path, validated_data, confidence, processed_text)
            except:
                pass  # キャッシュエラーは無視
            
            return validated_data, confidence, processed_text
            
        except Exception as e:
            self.logger.error(f"画像処理中にエラーが発生しました: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}, 0.0, ""
            
    def _is_vertical_text(self, image):
        """
        画像内のテキストが縦書きかどうかを検出する
        
        Args:
            image: 前処理済み画像
            
        Returns:
            bool: 縦書きの場合True、横書きの場合False
        """
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # ハフ変換で直線検出
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return False
            
        # 水平線と垂直線のカウント
        horizontal_count = 0
        vertical_count = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 線の角度を計算
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 水平線: 角度が0-30度または150-180度
            if angle < 30 or angle > 150:
                horizontal_count += 1
            # 垂直線: 角度が60-120度
            elif 60 < angle < 120:
                vertical_count += 1
                
        # 垂直線が水平線より多ければ縦書きと判断
        is_vertical = vertical_count > horizontal_count
        self.logger.debug(f"縦書き検出: 垂直線={vertical_count}, 水平線={horizontal_count}, 判定={is_vertical}")
        
        return is_vertical
        
    def _save_to_cache(self, image_path, data, confidence, text):
        """
        処理結果をキャッシュに保存する
        
        Args:
            image_path (str): 画像ファイルパス
            data (dict): 構造化データ
            confidence (float): 信頼度スコア
            text (str): 抽出テキスト
        """
        if not self.enable_cache:
            return
            
        try:
            # ファイル名からキャッシュキーを生成
            cache_key = hashlib.md5(image_path.encode()).hexdigest()
            
            # キャッシュデータの作成
            cache_data = {
                "image_path": image_path,
                "processed_at": datetime.datetime.now().isoformat(),
                "structured_data": data,
                "confidence": confidence,
                "text": text
            }
            
            # キャッシュディレクトリの確認と作成
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # キャッシュファイルに保存
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            self.logger.debug(f"キャッシュに保存しました: {cache_file}")
            
        except Exception as e:
            self.logger.warning(f"キャッシュへの保存に失敗しました: {str(e)}")

    def _postprocess_text(self, text, document_type='registry'):
        """
        OCRで認識されたテキストを後処理して品質を向上させる
        
        Args:
            text (str): OCRで認識されたテキスト
            document_type (str): 文書タイプ ('registry', 'certificate', 'mixed', 'vertical', 'other')
            
        Returns:
            str: 後処理されたテキスト
        """
        if not text:
            return text
            
        import re
        
        # 1. 余分な空白の削除・正規化
        text = re.sub(r'\s+', ' ', text)  # 連続する空白を1つに圧縮
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # 各行の先頭と末尾の空白を削除
        
        # 2. よくある誤認識の修正 (登記簿特有の文字修正)
        common_misrecognitions = {
            # 数字の誤認識
            'O': '0', 'o': '0', 'l': '1', 'I': '1', 'i': '1',
            'Z': '2', 'S': '5', 'b': '6', 'G': '6',
            # 記号の誤認識
            '，': ',', '．': '.', '、': ',', '。': '.',
            # 漢字の誤認識
            '土也': '地', '土田': '地', '已': '己', '巳': '己',
            '未': '来', '井': '非', '甲圧': '甲区', '乙圧': '乙区',
            # 区分所有・抵当権関連の誤認識
            '区分所宥': '区分所有', '区分斤有': '区分所有',
            '底当権': '抵当権', '氏当権': '抵当権', '氏当椎': '抵当権',
            # 登記固有用語
            '所有椎': '所有権', '所宥権': '所有権',
            '登言己': '登記', '登言己簿': '登記簿', '言己簿': '記簿',
            '順伯': '順位', '登言己原因': '登記原因',
            '権利者': '権利者', '継利者': '権利者',
            # 日付関連
            '令禾口': '令和', '令和口': '令和', '平咸': '平成'
        }
        
        for wrong, correct in common_misrecognitions.items():
            text = text.replace(wrong, correct)
            
        # 3. 文書タイプ別の処理
        if document_type in ['registry', 'mixed']:
            # 登記簿特有の後処理
            
            # 数値と単位の間の空白を削除 (例: 100 円 → 100円)
            text = re.sub(r'(\d+)\s+(円|平方メートル|㎡|m²)', r'\1\2', text)
            
            # 日付書式の正規化 (例: 令和 5 年 → 令和5年)
            text = re.sub(r'(平成|令和|昭和)\s*(\d{1,2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', 
                          r'\1\2年\3月\4日', text)
            
            # 甲区・乙区のフォーマット修正
            text = re.sub(r'甲\s*区', '甲区', text)
            text = re.sub(r'乙\s*区', '乙区', text)
            
            # 順位番号の修正 (例: 第 1 → 第1)
            text = re.sub(r'第\s+(\d+)', r'第\1', text)
            
            # 住所の修正 (例: 東京 都 → 東京都)
            prefecture_pattern = '(東京|大阪|京都|北海道|青森|岩手|宮城|秋田|山形|福島|茨城|栃木|群馬|埼玉|千葉|神奈川|新潟|富山|石川|福井|山梨|長野|岐阜|静岡|愛知|三重|滋賀|兵庫|奈良|和歌山|鳥取|島根|岡山|広島|山口|徳島|香川|愛媛|高知|福岡|佐賀|長崎|熊本|大分|宮崎|鹿児島|沖縄)'
            text = re.sub(f'{prefecture_pattern}\\s+都', r'\1都', text)
            text = re.sub(f'{prefecture_pattern}\\s+府', r'\1府', text)
            text = re.sub(f'{prefecture_pattern}\\s+県', r'\1県', text)
            
            # 住所の丁目・番地・号の修正
            text = re.sub(r'(\d+)\s*丁目', r'\1丁目', text)
            text = re.sub(r'(\d+)\s*番地?\s*(\d+)\s*号', r'\1番\2号', text)
            text = re.sub(r'(\d+)\s*番\s*(\d+)', r'\1番\2', text)
            
        elif document_type == 'certificate':
            # 権利証特有の後処理
            # 認証番号の修正 (例: 第 123456 号 → 第123456号)
            text = re.sub(r'第\s+([0-9A-Z]+)\s+号', r'第\1号', text)
            
        # 4. 一般的なテキスト修正
        
        # 括弧の誤認識修正
        text = re.sub(r'「', '(', text)
        text = re.sub(r'」', ')', text)
        
        # カタカナの修正 (濁点・半濁点の分離修正)
        katakana_daku = {
            'カ゛': 'ガ', 'キ゛': 'ギ', 'ク゛': 'グ', 'ケ゛': 'ゲ', 'コ゛': 'ゴ',
            'サ゛': 'ザ', 'シ゛': 'ジ', 'ス゛': 'ズ', 'セ゛': 'ゼ', 'ソ゛': 'ゾ',
            'タ゛': 'ダ', 'チ゛': 'ヂ', 'ツ゛': 'ヅ', 'テ゛': 'デ', 'ト゛': 'ド',
            'ハ゛': 'バ', 'ヒ゛': 'ビ', 'フ゛': 'ブ', 'ヘ゛': 'ベ', 'ホ゛': 'ボ',
            'ハ゜': 'パ', 'ヒ゜': 'ピ', 'フ゜': 'プ', 'ヘ゜': 'ペ', 'ホ゜': 'ポ'
        }
        for wrong, correct in katakana_daku.items():
            text = text.replace(wrong, correct)
        
        # 文字コード正規化 (半角カタカナを全角に、全角英数を半角に)
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # 5. ドキュメント構造を考慮した修正
        if document_type in ['registry', 'mixed']:
            # 登記事項証明書での表区切り文字の正規化
            text = re.sub(r'[-＝=─]{3,}', '─────────────────', text)
            
            # 抵当権・所有権などのセクション区切りの強調
            for section in ['所有権', '抵当権', '根抵当権', '地上権', '賃借権']:
                text = re.sub(f'({section})', f'\n\n{section}\n', text)
        
        self.logger.debug(f"テキスト後処理完了: {len(text)}文字")
        return text

    def _extract_text(self, image, orientation="vertical"):
        """
        画像からテキストを抽出する
        
        Args:
            image: 前処理済み画像
            orientation (str): テキストの向き ('vertical' または 'horizontal')
            
        Returns:
            str: 抽出されたテキスト
        """
        # Tesseractの設定を最適化
        config = self._optimize_tesseract_config(orientation)
        
        try:
            # 言語設定の取得
            lang = self.config.get('lang', 'jpn')
            
            # 縦書きテキストの場合、画像を回転
            if orientation == "vertical":
                # 画像を90度回転して縦書きテキストを横向きにする
                rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                
                # 日本語縦書きモードでOCR実行
                text = pytesseract.image_to_string(
                    rotated_image,
                    lang=lang,
                    config=config
                )
                
                # 縦書きテキストの場合、行の順序を右から左に修正
                # 日本語の縦書きは右から左に読むため
                lines = text.splitlines()
                if len(lines) > 1:
                    # 各行を配列に格納し逆順にする
                    reversed_lines = list(reversed(lines))
                    text = "\n".join(reversed_lines)
                    
            else:
                # 横書きテキストの場合、そのままOCR実行
                text = pytesseract.image_to_string(
                    image,
                    lang=lang,
                    config=config
                )
            
            # テキストが十分に抽出できなかった場合、別の設定で再試行
            if len(text.strip()) < 50:
                self.logger.warning("テキスト抽出結果が少なすぎます。別の設定で再試行します。")
                
                # 別の設定で再試行（セグメンテーションモードを変更）
                alt_config = f"{config} --psm 4" if orientation == "vertical" else f"{config} --psm 6"
                
                if orientation == "vertical":
                    text = pytesseract.image_to_string(
                        rotated_image,
                        lang=lang,
                        config=alt_config
                    )
                    lines = text.splitlines()
                    if len(lines) > 1:
                        reversed_lines = list(reversed(lines))
                        text = "\n".join(reversed_lines)
                else:
                    text = pytesseract.image_to_string(
                        image,
                        lang=lang,
                        config=alt_config
                    )
            
            # 空白行の削除と正規化
            text = "\n".join(line for line in text.splitlines() if line.strip())
            
            # テキスト抽出の結果をログに記録
            text_length = len(text)
            self.logger.info(f"テキスト抽出完了: {text_length}文字")
            if text_length < 100:
                self.logger.warning(f"抽出テキストが短すぎます: {text}")
                
            return text
            
        except Exception as e:
            self.logger.error(f"テキスト抽出エラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""

    def _preprocess_image(self, image_path):
        """
        OCR処理のための画像前処理を行う
        
        Args:
            image_path (str): 処理する画像ファイルのパス
            
        Returns:
            numpy.ndarray: 前処理済みの画像データ
            str: 検出されたテキスト向き ('horizontal' または 'vertical')
        """
        logger.info(f"画像の前処理を開始: {image_path}")
        
        try:
            # 画像の読み込み
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
            
            # 画像の前処理
            processed_img = self._process_image(img)
            
            # テキスト向きの検出
            orientation = self._detect_text_orientation(processed_img)
            
            logger.info(f"画像の前処理が完了しました: {image_path} (向き: {orientation})")
            return processed_img, orientation
            
        except Exception as e:
            logger.error(f"画像の前処理中にエラーが発生しました: {str(e)}", exc_info=True)
            # エラー時はオリジナル画像と水平向きを返す
            if 'img' in locals() and img is not None:
                return img, 'horizontal'
            else:
                # 読み込みに失敗した場合は空の画像を返す
                return np.zeros((100, 100, 3), dtype=np.uint8), 'horizontal'

    def _process_image(self, img):
        """
        画像の前処理を行う詳細なメソッド
        
        Args:
            img (numpy.ndarray): 処理する画像データ
            
        Returns:
            numpy.ndarray: 前処理済みの画像データ
        """
        preprocessing_params = {}
        img_enhanced = img.copy()
        
        # 画像のサイズ確認と調整
        if img.shape[0] < 1000 or img.shape[1] < 1000:
            scale_factor = max(1000 / img.shape[0], 1000 / img.shape[1])
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            img_enhanced = cv2.resize(img_enhanced, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            preprocessing_params['resized'] = True
            preprocessing_params['original_size'] = (img.shape[1], img.shape[0])
            preprocessing_params['new_size'] = (new_width, new_height)
        
        # グレースケールに変換
        if len(img_enhanced.shape) == 3:
            gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
            preprocessing_params['to_grayscale'] = True
        else:
            gray = img_enhanced
            preprocessing_params['to_grayscale'] = False
        
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        preprocessing_params['contrast_enhanced'] = True
        
        # 適応的な二値化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        preprocessing_params['adaptive_threshold'] = True
        
        # ノイズ除去（モルフォロジー演算）
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        preprocessing_params['noise_removal'] = True
        
        # 回転補正（スキューの検出と補正）
        # 登記簿は通常縦書きなので、縦方向のライン検出が重要
        is_vertical = self._detect_vertical_text(binary)
        preprocessing_params['is_vertical_text'] = is_vertical
        
        # 縦書きテキストの場合、回転補正を行う
        if is_vertical:
            # 必要に応じて画像の回転処理を実装
            # 縦書きテストのために90度回転した画像も作成
            rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
            preprocessing_params['rotated_for_vertical'] = True
        else:
            rotated = gray
            preprocessing_params['rotated_for_vertical'] = False
        
        # 余白のトリミング
        # テキスト領域を検出して余分な余白をトリミング
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # テキスト領域を含む最小の矩形を見つける
            x_min, y_min = img_enhanced.shape[1], img_enhanced.shape[0]
            x_max, y_max = 0, 0
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
            
            # 余白を追加して切り取り
            margin = 50
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(img_enhanced.shape[1], x_max + margin)
            y_max = min(img_enhanced.shape[0], y_max + margin)
            
            # 境界が有効な場合のみトリミング
            if x_min < x_max and y_min < y_max:
                img_enhanced = img_enhanced[y_min:y_max, x_min:x_max]
                preprocessing_params['trimmed'] = True
                preprocessing_params['trim_box'] = (x_min, y_min, x_max, y_max)
        
        # DPI情報の設定（Tesseractの精度向上のため）
        # これはPIL画像の属性として設定する必要がある場合がある
        preprocessing_params['dpi_set'] = True
        
        # ログに前処理パラメータを記録
        logger.debug(f"画像前処理パラメータ: {preprocessing_params}")
        
        return img_enhanced

    def _detect_text_orientation(self, img):
        """
        画像内のテキストの向き（縦書き/横書き）を検出する
        
        Args:
            img (numpy.ndarray): 分析する画像データ
            
        Returns:
            str: テキストの向き ('horizontal' または 'vertical')
        """
        # グレースケールに変換
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 二値化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 水平方向と垂直方向の射影プロファイルを計算
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        
        # 水平方向と垂直方向の変化量を計算
        h_diff = np.sum(np.abs(np.diff(h_proj)))
        v_diff = np.sum(np.abs(np.diff(v_proj)))
        
        # 登記簿に特有のパターンを探す（例: 「登記簿」「権利者」などの文字列）
        # この部分はテッセラクトを使って特定の領域のテキストを抽出し判断することも可能
        
        # 変化量の比率で判断
        if h_diff > v_diff * 1.2:
            # 水平方向の変化が大きい場合は縦書きの可能性が高い
            return 'vertical'
        else:
            # そうでない場合は横書きと判断
            return 'horizontal'

    def _detect_vertical_text(self, img):
        """
        画像内のテキストの向き（縦書き/横書き）を検出する
        
        Args:
            img (numpy.ndarray): 分析する画像データ
            
        Returns:
            bool: 縦書きの場合はTrue、横書きの場合はFalse
        """
        # グレースケールに変換
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 二値化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 水平方向と垂直方向の射影プロファイルを計算
        h_proj = np.sum(binary, axis=1)
        v_proj = np.sum(binary, axis=0)
        
        # 水平方向と垂直方向の変化量を計算
        h_diff = np.sum(np.abs(np.diff(h_proj)))
        v_diff = np.sum(np.abs(np.diff(v_proj)))
        
        # 変化量の比率で判断
        if h_diff > v_diff * 1.2:
            # 水平方向の変化が大きい場合は縦書きの可能性が高い
            return True
        else:
            # そうでない場合は横書きと判断
            return False

    def _calculate_confidence(self, structured_data):
        """
        抽出された構造化データの信頼度を計算する
        
        Args:
            structured_data (dict): 構造化データ
            
        Returns:
            float: 0.0～1.0の信頼度スコア
        """
        if not structured_data:
            return 0.0
        
        # 重要フィールドとその重み
        field_weights = {
            'property_number': 0.15,  # 不動産番号
            'address': 0.15,          # 所在
            'lot_number': 0.10,       # 地番
            'area': 0.10,             # 地積
            'owner_name': 0.20,       # 所有者名
            'owner_address': 0.15,    # 所有者住所
            'registration_date': 0.10  # 登記日付
        }
        
        # 存在・形式の正確性に基づいて各フィールドの信頼度を評価
        field_scores = {}
        
        # 不動産番号の評価
        if 'property_number' in structured_data and structured_data['property_number']:
            property_number = structured_data['property_number']
            import re
            if re.match(r'^\d{4}-\d{4}-\d{4}$', property_number):
                field_scores['property_number'] = 1.0  # 完全な形式
            elif re.match(r'^\d{12}$', property_number):
                field_scores['property_number'] = 0.9  # ハイフンがない
            elif re.match(r'^\d{4}[-]?\d{4}[-]?\d{1,3}', property_number):
                field_scores['property_number'] = 0.7  # 一部欠損
            else:
                field_scores['property_number'] = 0.3  # 形式が異なる
        else:
            field_scores['property_number'] = 0.0
        
        # 住所の評価
        if 'address' in structured_data and structured_data['address']:
            address = structured_data['address']
            if len(address) > 8:  # 十分な長さの住所
                # 一般的な都道府県名が含まれるか
                import re
                prefectures = '東京都|北海道|大阪府|京都府|神奈川県|埼玉県|千葉県|愛知県|兵庫県|福岡県'
                if re.search(f'({prefectures})', address):
                    field_scores['address'] = 0.9
                else:
                    field_scores['address'] = 0.6
            else:
                field_scores['address'] = 0.3  # 住所が短すぎる
        else:
            field_scores['address'] = 0.0
        
        # 地番の評価
        if 'lot_number' in structured_data and structured_data['lot_number']:
            lot_number = structured_data['lot_number']
            import re
            if re.match(r'^\d+(-\d+)?$', lot_number):
                field_scores['lot_number'] = 1.0  # 数字とハイフンのみ
            elif re.match(r'^\d+番\d*号?$', lot_number):
                field_scores['lot_number'] = 0.9  # 「番」「号」を含む
            elif re.search(r'\d+', lot_number):
                field_scores['lot_number'] = 0.7  # 数字を含む
            else:
                field_scores['lot_number'] = 0.4
        else:
            field_scores['lot_number'] = 0.0
        
        # 地積（面積）の評価
        if 'area' in structured_data:
            area = structured_data['area']
            if isinstance(area, (int, float)) and area > 0:
                field_scores['area'] = 1.0  # 数値型で正の値
            elif isinstance(area, str) and re.search(r'\d+\.?\d*', area):
                field_scores['area'] = 0.7  # 数字を含む文字列
            else:
                field_scores['area'] = 0.3
        else:
            field_scores['area'] = 0.0
        
        # 所有者名の評価
        if 'owner_name' in structured_data and structured_data['owner_name']:
            owner_name = structured_data['owner_name']
            if len(owner_name) >= 2:
                # 一般的な名前の特徴（個人名または法人名）
                import re
                if re.search(r'株式会社|有限会社|合同会社', owner_name):
                    field_scores['owner_name'] = 0.9  # 法人名
                elif len(owner_name) <= 20:  # 個人名は比較的短い
                    field_scores['owner_name'] = 0.8
                else:
                    field_scores['owner_name'] = 0.6
            else:
                field_scores['owner_name'] = 0.3  # 名前が短すぎる
        else:
            field_scores['owner_name'] = 0.0
        
        # 所有者住所の評価
        if 'owner_address' in structured_data and structured_data['owner_address']:
            owner_address = structured_data['owner_address']
            if len(owner_address) > 8:
                import re
                prefectures = '東京都|北海道|大阪府|京都府|神奈川県|埼玉県|千葉県|愛知県|兵庫県|福岡県'
                if re.search(f'({prefectures})', owner_address):
                    field_scores['owner_address'] = 0.9
                else:
                    field_scores['owner_address'] = 0.6
            else:
                field_scores['owner_address'] = 0.3
        else:
            field_scores['owner_address'] = 0.0
        
        # 登記日付の評価
        if 'registration_date' in structured_data and structured_data['registration_date']:
            date_str = structured_data['registration_date']
            import re
            # YYYY/MM/DD形式または和暦形式
            if re.match(r'^\d{4}/\d{1,2}/\d{1,2}$', date_str):
                field_scores['registration_date'] = 1.0
            elif re.search(r'(平成|令和|昭和)\d{1,2}年\d{1,2}月\d{1,2}日', date_str):
                field_scores['registration_date'] = 0.9
            elif re.search(r'\d{1,2}年\d{1,2}月\d{1,2}日', date_str):
                field_scores['registration_date'] = 0.7
            elif re.search(r'\d{1,2}/\d{1,2}/\d{1,4}', date_str):
                field_scores['registration_date'] = 0.7
            else:
                field_scores['registration_date'] = 0.4
        else:
            field_scores['registration_date'] = 0.0
        
        # 全体の信頼度を計算（重み付け合計）
        total_weight = 0
        weighted_score = 0
        
        for field, weight in field_weights.items():
            if field in field_scores:
                weighted_score += field_scores[field] * weight
                total_weight += weight
        
        # 重みの合計が0でない場合のみ計算
        if total_weight > 0:
            overall_confidence = weighted_score / total_weight
        else:
            overall_confidence = 0.0
        
        # 抵当権情報があれば追加のボーナス
        if 'mortgage_rights' in structured_data and structured_data['mortgage_rights']:
            mortgage_count = len(structured_data['mortgage_rights'])
            if mortgage_count > 0:
                # 抵当権情報があると信頼度が上がる（最大0.1のボーナス）
                mortgage_bonus = min(mortgage_count * 0.02, 0.1)
                overall_confidence = min(overall_confidence + mortgage_bonus, 1.0)
        
        # デバッグ情報
        self.logger.debug(f"フィールド信頼度: {field_scores}")
        self.logger.debug(f"全体信頼度: {overall_confidence:.2f}")
        
        return overall_confidence

    def _check_dependencies(self):
        """
        必要なライブラリの存在確認を行う
        """
        # このメソッドの実装は必要なライブラリの存在確認を行う
        # 実際の実装は環境に応じて適切に実装する必要がある
        pass

    def extract_structured_data(self, text):
        """
        OCRテキストから構造化データを抽出する
        
        Args:
            text (str): OCRで抽出されたテキスト
            
        Returns:
            dict: 構造化されたデータ
        """
        self.logger.info("構造化データの抽出を開始")
        
        # 抵当権情報の抽出
        mortgage_rights = self._extract_mortgage_rights(text)
        
        # 所有権情報の抽出
        ownership_rights = self._extract_ownership_rights(text)
        
        # 構造化データを統合
        structured_data = {**ownership_rights, "mortgage_rights": mortgage_rights}
        
        # 空の値をクリーンアップ
        cleaned_data = self._clean_empty_values(structured_data)
        
        # データの後処理と正規化
        processed_data = self.postprocess_ocr_text(cleaned_data)
        
        # 信頼度スコアを計算
        confidence = self._calculate_confidence(processed_data)
        
        self.logger.info(f"構造化データの抽出完了: {len(processed_data)} フィールド, 信頼度: {confidence:.2f}")
        
        return processed_data, confidence

    def _extract_mortgage_rights(self, ocr_text):
        """
        OCRテキストから抵当権情報を抽出する
        
        Args:
            ocr_text (str): OCRで抽出されたテキスト
            
        Returns:
            list: 抵当権情報のリスト（辞書形式）
        """
        if not ocr_text:
            return []
            
        mortgage_rights = []
        
        # 乙区（抵当権）のセクションを検出
        mortgage_section_pattern = r"乙区.*?((?:順位|順位番号).*?(?:権利者|債権者).*?)(?:丙区|$)"
        mortgage_section_match = re.search(mortgage_section_pattern, ocr_text, re.DOTALL | re.IGNORECASE)
        
        if not mortgage_section_match:
            return []
            
        mortgage_section = mortgage_section_match.group(1)
        
        # 順位番号ごとに分割
        rank_entries = re.split(r"(?:順位|順位番号)\s*(\d+)", mortgage_section)
        
        if len(rank_entries) <= 1:
            return []
            
        # 最初のエントリーは分割の前の部分なので無視
        for i in range(1, len(rank_entries), 2):
            if i + 1 >= len(rank_entries):
                break
                
            rank = rank_entries[i]
            entry = rank_entries[i + 1]
            
            # 抵当権の目的・種類を抽出
            purpose_pattern = r"(?:目的|権利の種類)[：:]\s*(.+?)(?:\n|$)"
            purpose_match = re.search(purpose_pattern, entry)
            purpose = purpose_match.group(1).strip() if purpose_match else ""
            
            # 日付を抽出（和暦・西暦両方対応）
            date_pattern = r"(?:日付|受付日(?:時)?)[：:]\s*(?:令和|平成|昭和)?\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日"
            date_match = re.search(date_pattern, entry)
            
            registration_date = None
            if date_match:
                year, month, day = date_match.groups()
                # 和暦の場合は西暦に変換する処理を追加（この例では単純化）
                era_pattern = r"(令和|平成|昭和)\s*(\d+)\s*年"
                era_match = re.search(era_pattern, entry)
                if era_match:
                    era, year = era_match.groups()
                    year = int(year)
                    if era == "令和":
                        year += 2018
                    elif era == "平成":
                        year += 1988
                    elif era == "昭和":
                        year += 1925
                registration_date = f"{year}年{month}月{day}日"
            
            # 原因を抽出
            cause_pattern = r"(?:原因|登記原因)[：:]\s*(.+?)(?:\n|$)"
            cause_match = re.search(cause_pattern, entry)
            cause = cause_match.group(1).strip() if cause_match else ""
            
            # 金額を抽出
            amount_pattern = r"(?:金額|債権額(?:等)?)[：:]\s*(.+?)(?:\n|$)"
            amount_match = re.search(amount_pattern, entry)
            amount = amount_match.group(1).strip() if amount_match else ""
            
            # 利率を抽出
            interest_pattern = r"(?:利率|利息)[：:]\s*(.+?)(?:\n|$)"
            interest_match = re.search(interest_pattern, entry)
            interest_rate = interest_match.group(1).strip() if interest_match else ""
            
            # 債権者を抽出
            creditor_pattern = r"(?:債権者|権利者)[：:]\s*(.+?)(?:\n|$)"
            creditor_match = re.search(creditor_pattern, entry)
            creditor = creditor_match.group(1).strip() if creditor_match else ""
            
            # 債務者を抽出
            debtor_pattern = r"(?:債務者)[：:]\s*(.+?)(?:\n|$)"
            debtor_match = re.search(debtor_pattern, entry)
            debtor = debtor_match.group(1).strip() if debtor_match else ""
            
            mortgage_right = {
                "section": "乙区",
                "rank": rank,
                "type": "抵当権",
                "purpose": purpose,
                "date": registration_date,
                "cause": cause,
                "amount": amount,
                "interest_rate": interest_rate,
                "creditor": creditor,
                "debtor": debtor
            }
            
            mortgage_rights.append(mortgage_right)
            
        return mortgage_rights

    def _extract_ownership_rights(self, ocr_text):
        """
        OCRテキストから所有権情報を抽出する
        
        Args:
            ocr_text (str): OCRで抽出されたテキスト
            
        Returns:
            list: 所有権情報のリスト（辞書形式）
        """
        if not ocr_text:
            return []
            
        ownership_rights = []
        
        # 甲区（所有権）のセクションを検出
        ownership_section_pattern = r"甲区.*?((?:順位|順位番号).*?(?:権利者|所有者).*?)(?:乙区|$)"
        ownership_section_match = re.search(ownership_section_pattern, ocr_text, re.DOTALL | re.IGNORECASE)
        
        if not ownership_section_match:
            return []
            
        ownership_section = ownership_section_match.group(1)
        
        # 順位番号ごとに分割
        rank_entries = re.split(r"(?:順位|順位番号)\s*(\d+)", ownership_section)
        
        if len(rank_entries) <= 1:
            return []
            
        # 最初のエントリーは分割の前の部分なので無視
        for i in range(1, len(rank_entries), 2):
            if i + 1 >= len(rank_entries):
                break
                
            rank = rank_entries[i]
            entry = rank_entries[i + 1]
            
            # 所有権の目的・種類を抽出
            purpose_pattern = r"(?:目的|権利の種類)[：:]\s*(.+?)(?:\n|$)"
            purpose_match = re.search(purpose_pattern, entry)
            purpose = purpose_match.group(1).strip() if purpose_match else "所有権"
            
            # 日付を抽出（和暦・西暦両方対応）
            date_pattern = r"(?:日付|受付日(?:時)?)[：:]\s*(?:令和|平成|昭和)?\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日"
            date_match = re.search(date_pattern, entry)
            
            registration_date = None
            if date_match:
                year, month, day = date_match.groups()
                # 和暦の場合は西暦に変換する処理を追加
                era_pattern = r"(令和|平成|昭和)\s*(\d+)\s*年"
                era_match = re.search(era_pattern, entry)
                if era_match:
                    era, year = era_match.groups()
                    year = int(year)
                    if era == "令和":
                        year += 2018
                    elif era == "平成":
                        year += 1988
                    elif era == "昭和":
                        year += 1925
                registration_date = f"{year}年{month}月{day}日"
            
            # 原因を抽出
            cause_pattern = r"(?:原因|登記原因)[：:]\s*(.+?)(?:\n|$)"
            cause_match = re.search(cause_pattern, entry)
            cause = cause_match.group(1).strip() if cause_match else ""
            
            # 所有者（権利者）を抽出
            owner_pattern = r"(?:所有者|権利者)[：:]\s*(.+?)(?:\n|所有者住所|権利者住所|$)"
            owner_match = re.search(owner_pattern, entry, re.DOTALL)
            owner = owner_match.group(1).strip() if owner_match else ""
            
            # 所有者住所を抽出
            address_pattern = r"(?:所有者住所|権利者住所)[：:]\s*(.+?)(?:\n|$)"
            address_match = re.search(address_pattern, entry, re.DOTALL)
            address = address_match.group(1).strip() if address_match else ""
            
            # 住所が所有者名に含まれている場合の処理
            if not address and owner:
                address_in_owner = re.search(r"(.+?)(?:住所[：:]|所在地[：:]|在住[：:])\s*(.+)", owner)
                if address_in_owner:
                    owner = address_in_owner.group(1).strip()
                    address = address_in_owner.group(2).strip()
            
            ownership_right = {
                "section": "甲区",
                "rank": rank,
                "type": "所有権",
                "purpose": purpose,
                "date": registration_date,
                "cause": cause,
                "owner": owner,
                "owner_address": address
            }
            
            ownership_rights.append(ownership_right)
            
        return ownership_rights

    def enhance_image(self, image_path):
        """
        画像の前処理を行い、OCR精度を向上させる
        
        Args:
            image_path (str): 処理する画像のパス
        
        Returns:
            str: 前処理後の画像のパス
        """
        logger.info("画像前処理を開始: %s", image_path)
        
        try:
            # 画像読み込み
            img = cv2.imread(image_path)
            if img is None:
                logger.error("画像の読み込みに失敗しました: %s", image_path)
                return image_path
            
            # グレースケール変換
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # ノイズ除去（バイラテラルフィルタ）
            denoise = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 適応的二値化
            thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # 傾き補正（自動方向検出が有効な場合）
            if self.config['detect_orientation']:
                # エッジを検出してから傾き推定を行う
                edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
                
                if lines is not None and len(lines) > 0:
                    angles = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if x2 - x1 != 0:  # 垂直線を避ける
                            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                            angles.append(angle)
                    
                    if angles:
                        # 最も頻度の高い角度を見つける
                        from collections import Counter
                        angle_counts = Counter([round(a) for a in angles])
                        dominant_angle = angle_counts.most_common(1)[0][0]
                        
                        # 画像を回転
                        if abs(dominant_angle) > 0.5:  # 角度が小さすぎる場合は回転しない
                            (h, w) = thresh.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, dominant_angle, 1.0)
                            thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, 
                                                borderMode=cv2.BORDER_REPLICATE)
                            logger.info("画像を %f 度回転しました", dominant_angle)
            
            # コントラスト強化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(thresh)
            
            # 線の除去（登記簿の枠線などを除去）
            if self.config['remove_lines']:
                # 水平線と垂直線を検出して除去
                horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
                vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
                
                # 水平線を検出
                horizontal_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
                # 垂直線を検出
                vertical_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
                
                # 検出した線を結合
                lines = horizontal_lines + vertical_lines
                
                # 線を除去（白黒反転して計算）
                enhanced = cv2.bitwise_not(enhanced)
                lines = cv2.bitwise_not(lines)
                no_lines = cv2.bitwise_and(enhanced, enhanced, mask=lines)
                enhanced = cv2.bitwise_not(no_lines)
            
            # 拡大（解像度向上）
            scale_percent = 200
            width = int(enhanced.shape[1] * scale_percent / 100)
            height = int(enhanced.shape[0] * scale_percent / 100)
            enlarged = cv2.resize(enhanced, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # 結果を保存
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.temp_dir, f"{name}_enhanced{ext}")
            cv2.imwrite(output_path, enlarged)
            
            # デバッグ用に前処理画像も保存
            debug_path = os.path.join(self.temp_dir, "debug_preprocess.png")
            cv2.imwrite(debug_path, enlarged)
            
            logger.info("画像前処理が完了しました: %s", output_path)
            return output_path
            
        except Exception as e:
            logger.error("画像前処理中にエラーが発生しました: %s", str(e), exc_info=True)
            return image_path  # エラーが発生した場合は元の画像を返す

    def postprocess_ocr_text(self, structured_data):
        """
        OCR結果の構造化データを後処理して品質を向上させる
        
        Args:
            structured_data (dict): OCRで抽出された構造化データ
            
        Returns:
            dict: 後処理された構造化データ
        """
        if not structured_data:
            return structured_data
        
        try:
            # 不動産番号の正規化（数字とハイフンのみ）
            if 'property_number' in structured_data:
                property_number = structured_data['property_number']
                if property_number:
                    # 数字とハイフンのみ抽出
                    import re
                    clean_number = ''.join(c for c in property_number if c.isdigit() or c == '-')
                    # 形式チェック（例：1234-5678-9012）
                    if re.match(r'^\d{4}-\d{4}-\d{4}$', clean_number):
                        structured_data['property_number'] = clean_number
                    elif re.match(r'^\d{12}$', clean_number):
                        # ハイフンがない場合、適切な位置に挿入
                        formatted_number = f"{clean_number[:4]}-{clean_number[4:8]}-{clean_number[8:]}"
                        structured_data['property_number'] = formatted_number
            
            # 地積（面積）の数値化
            if 'area' in structured_data:
                area_str = structured_data['area']
                if isinstance(area_str, str) and area_str:
                    # カンマ、空白を除去
                    clean_area = area_str.replace(',', '').replace(' ', '')
                    # 数字と小数点のみ抽出
                    import re
                    clean_area = re.sub(r'[^\d.]', '', clean_area)
                    try:
                        structured_data['area'] = float(clean_area)
                    except:
                        pass  # 変換できない場合は元の値を維持
            
            # 所有者名のフォーマット統一
            if 'owner_name' in structured_data and structured_data['owner_name']:
                owner_name = structured_data['owner_name']
                # 余分な空白を削除
                owner_name = ' '.join(owner_name.split())
                # 氏名の一般的なパターンを修正
                import re
                # "株式会社XXX" のパターンを修正
                if re.search(r'株式会朶|株式会社|株式台社', owner_name):
                    owner_name = re.sub(r'株式会朶|株式台社', '株式会社', owner_name)
                structured_data['owner_name'] = owner_name
            
            # 住所の正規化
            if 'address' in structured_data and structured_data['address']:
                address = structured_data['address']
                # 余分な空白を削除
                address = ' '.join(address.split())
                # 誤認識されやすい住所パターンを修正
                import re
                address = re.sub(r'T\s*県', '千葉県', address)
                address = re.sub(r'神奈jll', '神奈川', address)
                structured_data['address'] = address
            
            # 日付の正規化
            if 'registration_date' in structured_data and structured_data['registration_date']:
                date_str = structured_data['registration_date']
                # 日付のフォーマット統一（YYYY/MM/DD）
                import re
                # 和暦から西暦への変換
                match = re.search(r'(平成|令和|昭和)(\d{1,2})年(\d{1,2})月(\d{1,2})日', date_str)
                if match:
                    era, year, month, day = match.groups()
                    year = int(year)
                    if era == '平成':
                        year += 1988  # 平成元年 = 1989年
                    elif era == '令和':
                        year += 2018  # 令和元年 = 2019年
                    elif era == '昭和':
                        year += 1925  # 昭和元年 = 1926年
                    
                    # YYYY/MM/DD形式に変換
                    structured_data['registration_date'] = f"{year}/{int(month)}/{int(day)}"
            
            return structured_data
            
        except Exception as e:
            self.logger.error(f"構造化データの後処理中にエラーが発生しました: {str(e)}")
            return structured_data  # エラーが発生した場合は元のデータを返す

    def hybrid_ocr(self, image_path):
        """
        複数のOCRエンジンを組み合わせた複合アプローチでOCRを実行
        
        Args:
            image_path (str): 処理する画像のパス
            
        Returns:
            tuple: (構造化データ辞書, 信頼度スコア, 抽出テキスト)
        """
        self.logger.info(f"ハイブリッドOCR処理開始: {image_path}")
        
        # 結果を保存する辞書
        results = {}
        
        # 1. 標準のOCR処理（TesseractベースのPyTesseract）
        try:
            self.logger.info("標準OCR処理を実行中...")
            standard_data, standard_confidence, standard_text = self.process_image(image_path)
            results['standard'] = {
                'data': standard_data,
                'confidence': standard_confidence,
                'text': standard_text
            }
            self.logger.info(f"標準OCR処理完了: 信頼度 {standard_confidence:.2f}")
        except Exception as e:
            self.logger.error(f"標準OCR処理でエラー: {str(e)}")
            results['standard'] = {
                'data': {},
                'confidence': 0.0,
                'text': ""
            }
        
        # 2. 日本語特化OCR処理（PSMを変更）
        try:
            self.logger.info("日本語特化OCR処理を実行中...")
            # 設定をコピーして日本語特化用に調整
            jp_config = self.config.copy()
            jp_config['psm'] = 4  # 縦書きテキスト向けのPSM
            jp_config['optimize_japanese'] = True
            
            # 一時的に設定を変更
            original_config = self.config.copy()
            self.config = jp_config
            
            # 日本語特化OCR実行
            jp_data, jp_confidence, jp_text = self.process_image(image_path)
            results['japanese'] = {
                'data': jp_data,
                'confidence': jp_confidence,
                'text': jp_text
            }
            
            # 設定を元に戻す
            self.config = original_config
            self.logger.info(f"日本語特化OCR処理完了: 信頼度 {jp_confidence:.2f}")
        except Exception as e:
            self.logger.error(f"日本語特化OCR処理でエラー: {str(e)}")
            results['japanese'] = {
                'data': {},
                'confidence': 0.0,
                'text': ""
            }
        
        # 3. 高精度OCR処理（高DPIと前処理強化）
        try:
            self.logger.info("高精度OCR処理を実行中...")
            # 設定をコピーして高精度用に調整
            hd_config = self.config.copy()
            hd_config['dpi'] = 600  # 高DPI
            hd_config['enhance_preprocessing'] = True
            hd_config['remove_lines'] = True
            
            # 一時的に設定を変更
            original_config = self.config.copy()
            self.config = hd_config
            
            # 高精度OCR実行
            hd_data, hd_confidence, hd_text = self.process_image(image_path)
            results['high_quality'] = {
                'data': hd_data,
                'confidence': hd_confidence,
                'text': hd_text
            }
            
            # 設定を元に戻す
            self.config = original_config
            self.logger.info(f"高精度OCR処理完了: 信頼度 {hd_confidence:.2f}")
        except Exception as e:
            self.logger.error(f"高精度OCR処理でエラー: {str(e)}")
            results['high_quality'] = {
                'data': {},
                'confidence': 0.0,
                'text': ""
            }
        
        # 4. OpenAI OCR処理（もし利用可能であれば）
        openai_available = False
        try:
            from openai_ocr import OpenAIOCR
            openai_available = True
        except ImportError:
            openai_available = False
        
        if openai_available and os.environ.get('OPENAI_API_KEY'):
            try:
                self.logger.info("OpenAI Vision OCR処理を実行中...")
                from openai_ocr import OpenAIOCR
                
                # OpenAI OCRの設定
                openai_config = {
                    'api_key': os.environ.get('OPENAI_API_KEY'),
                    'model': 'gpt-4-vision-preview',
                    'temperature': 0.3,
                    'max_tokens': 4000,
                    'detail_level': 'high'
                }
                
                # OpenAI OCRエンジンの初期化と実行
                openai_ocr = OpenAIOCR(openai_config)
                openai_data, openai_confidence, openai_text = openai_ocr.process_image(image_path)
                
                results['openai'] = {
                    'data': openai_data,
                    'confidence': openai_confidence,
                    'text': openai_text
                }
                self.logger.info(f"OpenAI Vision OCR処理完了: 信頼度 {openai_confidence:.2f}")
            except Exception as e:
                self.logger.error(f"OpenAI Vision OCR処理でエラー: {str(e)}")
                results['openai'] = {
                    'data': {},
                    'confidence': 0.0,
                    'text': ""
                }
        
        # 最終結果の統合（最も高い信頼度の結果をベースに）
        best_engine = None
        best_confidence = 0.0
        
        for engine, data in results.items():
            if data['confidence'] > best_confidence:
                best_confidence = data['confidence']
                best_engine = engine
        
        if best_engine is None:
            self.logger.warning("有効なOCR結果が見つかりませんでした")
            return {}, 0.0, ""
        
        self.logger.info(f"最良の結果: {best_engine} (信頼度: {best_confidence:.2f})")
        
        # ベースとなる結果
        base_result = results[best_engine]['data']
        base_text = results[best_engine]['text']
        
        # 各フィールドで最も信頼性の高い値を採用してマージ
        important_fields = ['property_number', 'address', 'lot_number', 'area', 'owner_name', 'owner_address', 'registration_date']
        merged_result = base_result.copy()
        
        for field in important_fields:
            best_field_value = None
            best_field_confidence = 0.0
            
            for engine, data in results.items():
                engine_data = data['data']
                engine_confidence = data['confidence']
                
                if field in engine_data and engine_data[field]:
                    # フィールド固有の信頼度評価（一時的なもの）
                    field_confidence = engine_confidence
                    
                    # 特定フィールドの評価を調整
                    if field == 'property_number' and engine == 'high_quality':
                        field_confidence *= 1.2  # 高精度処理は不動産番号に強い
                    elif field == 'registration_date' and engine == 'japanese':
                        field_confidence *= 1.1  # 日本語処理は日付に強い
                    elif field == 'owner_name' and engine == 'openai':
                        field_confidence *= 1.3  # OpenAIは名前認識に強い
                    
                    if field_confidence > best_field_confidence:
                        best_field_confidence = field_confidence
                        best_field_value = engine_data[field]
            
            if best_field_value:
                merged_result[field] = best_field_value
        
        # 抵当権情報の統合（最も詳細なものを採用）
        if 'mortgage_rights' in base_result:
            best_mortgage_rights = base_result['mortgage_rights']
            best_mortgage_count = len(best_mortgage_rights)
            
            for engine, data in results.items():
                if engine == best_engine:
                    continue
                
                engine_data = data['data']
                if 'mortgage_rights' in engine_data:
                    engine_mortgage_rights = engine_data['mortgage_rights']
                    engine_mortgage_count = len(engine_mortgage_rights)
                    
                    if engine_mortgage_count > best_mortgage_count:
                        best_mortgage_rights = engine_mortgage_rights
                        best_mortgage_count = engine_mortgage_count
            
            merged_result['mortgage_rights'] = best_mortgage_rights
        
        # 最終結果の信頼度を再計算
        final_confidence = self._calculate_confidence(merged_result)
        
        self.logger.info(f"ハイブリッドOCR処理完了: 最終信頼度 {final_confidence:.2f}")
        
        return merged_result, final_confidence, base_text
