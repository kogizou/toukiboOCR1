"""
登記簿OCRアプリ - メインアプリケーション

このスクリプトは登記簿OCRアプリのメインアプリケーションを実装します。
Flask Webフレームワークを使用して、ユーザーインターフェースとOCR機能を統合します。
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
import pandas as pd
import re
import sqlite3

# OCR実装モジュールをインポート
from ocr_implementation import RegistryOCR, logger as ocr_logger
# データベースモジュールをインポート
from database import OCRDatabase, logger as db_logger
# ユーザー認証モジュールをインポート
from auth import UserAuth, login_required, admin_required, logger as auth_logger

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('RegistryOCRApp')

# Flaskアプリケーションの初期化
app = Flask(__name__)
app.secret_key = 'tokibo_ocr_secret_key'  # セッション用の秘密鍵

# アプリケーション設定
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
app.config['TEMP_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 最大32MB

# 必要なディレクトリの作成
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['TEMP_FOLDER'], exist_ok=True)

# OCRエンジンの初期化
ocr_config = {
    'temp_dir': app.config['TEMP_FOLDER'],
    'output_dir': app.config['OUTPUT_FOLDER'],
    'dpi': 300,
    'psm': 6,
    'oem': 3
}
ocr_engine = RegistryOCR(ocr_config)

# データベースの初期化
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'registry_ocr.db')
db = OCRDatabase(db_path)
logger.info("Database initialized at: %s", db_path)

# ユーザー認証の初期化
auth = UserAuth(db_path)
logger.info("User authentication initialized")

# デフォルト設定のロード
if not os.path.exists('config.json'):
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(ocr_config, f, ensure_ascii=False, indent=2)
    logger.info("Created default config file: config.json")
else:
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
            ocr_config.update(loaded_config)
            # OCRエンジンの設定を更新
            ocr_engine.config.update(ocr_config)
        logger.info("Loaded config from config.json")
    except Exception as e:
        logger.error("Failed to load config.json: %s", str(e))

# ファイル拡張子の確認
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ダッシュボード画面
@app.route('/')
@login_required
def dashboard():
    # 最近処理されたファイルの取得（最新5件）
    recent_files = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith('.json')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['OUTPUT_FOLDER'], x)), reverse=True)
        
        for file in files[:5]:
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    recent_files.append({
                        'filename': data.get('file_name', 'Unknown'),
                        'processed_at': data.get('processed_at', ''),
                        'confidence': round(data.get('confidence', 0), 1),
                        'result_file': file
                    })
            except Exception as e:
                logger.error("Error reading file %s: %s", file, str(e))
    
    # 統計情報の取得
    stats = {
        'total_files': len([f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith('.json')]) if os.path.exists(app.config['OUTPUT_FOLDER']) else 0,
        'pending_files': len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]) if os.path.exists(app.config['UPLOAD_FOLDER']) else 0,
        'avg_confidence': 0
    }
    
    # 平均信頼度の計算
    if recent_files:
        avg_confidence = sum(file['confidence'] for file in recent_files) / len(recent_files)
        stats['avg_confidence'] = round(avg_confidence, 1)
    
    return render_template('dashboard.html', recent_files=recent_files, stats=stats)

# スキャン・OCR画面
@app.route('/scan', methods=['GET', 'POST'])
@login_required
def scan():
    if request.method == 'POST':
        # OCR設定の取得
        ocr_settings = {
            'dpi': int(request.form.get('dpi', 300)),
            'psm': int(request.form.get('psm', 6)),
            'oem': int(request.form.get('oem', 3))
        }
        
        # OCRエンジンの設定を更新
        ocr_engine.config.update(ocr_settings)
        
        # ファイルがリクエストに含まれているか確認
        if 'file' not in request.files:
            flash('ファイルが選択されていません')
            return redirect(request.url)
        
        file = request.files['file']
        
        # ファイル名が空でないか確認
        if file.filename == '':
            flash('ファイルが選択されていません')
            return redirect(request.url)
        
        # 有効なファイルか確認してアップロード
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                logger.info("Starting OCR process for file: %s", filename)
                # OCR処理の実行
                result = ocr_engine.process_image(file_path)
                output_path = ocr_engine.save_result(result)
                
                # データベースに結果を保存
                db.save_ocr_result(result)
                logger.info("OCR result saved to database")
                
                # 処理結果ページにリダイレクト
                result_filename = os.path.basename(output_path)
                logger.info("OCR process completed. Result saved to: %s", result_filename)
                return redirect(url_for('result', filename=result_filename))
            
            except Exception as e:
                logger.error("OCR process failed: %s", str(e), exc_info=True)
                flash(f'OCR処理中にエラーが発生しました: {str(e)}')
                return redirect(request.url)
        else:
            flash('許可されていないファイル形式です')
            return redirect(request.url)
    
    # OCR設定を表示
    ocr_settings = {
        'dpi': ocr_engine.config.get('dpi', 300),
        'psm': ocr_engine.config.get('psm', 6),
        'oem': ocr_engine.config.get('oem', 3)
    }
    
    # PSMオプションの定義
    psm_options = {
        0: 'オリエンテーションと単語の自動検出（OSD）のみ',
        1: '自動ページセグメンテーションとOSD',
        2: '自動ページセグメンテーション（OSDなし、OCRなし）',
        3: '完全自動ページセグメンテーション（OSDなし）',
        4: '可変サイズの単一列として仮定',
        5: '均一なテキストブロックとして仮定',
        6: '均一なテキストブロックとして仮定（単一の均一なブロック）',
        7: '画像を単一のテキスト行として扱う',
        8: '画像を単一の単語として扱う',
        9: '画像を円の中の単一の単語として扱う',
        10: '画像を単一の文字として扱う',
        11: '疎なテキスト。あらゆる方向と順序でできるだけ多くのテキストを見つける',
        12: '疎なテキストと強制的なOSD',
        13: '生のライン。画像を単一のテキスト行として扱う（ヒューリスティックなモデル無効）'
    }
    
    # OEMオプションの定義
    oem_options = {
        0: 'レガシーエンジンのみ',
        1: 'ニューラルネットワークLSTMエンジンのみ',
        2: 'レガシー + LSTMエンジン',
        3: 'デフォルト（利用可能な最適なエンジン）'
    }
    
    return render_template('scan.html', ocr_settings=ocr_settings, psm_options=psm_options, oem_options=oem_options)

# 処理結果表示画面
@app.route('/result/<filename>')
def result(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash('指定されたファイルが見つかりません')
        return redirect(url_for('dashboard'))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
        
        # 元の画像ファイル名
        original_filename = result_data.get('file_name', 'Unknown')
        
        # 構造化データ
        structured_data = result_data.get('structured_data', {})
        
        # 処理日時のフォーマット
        processed_at = result_data.get('processed_at', '')
        try:
            dt = datetime.fromisoformat(processed_at)
            formatted_date = dt.strftime('%Y年%m月%d日 %H:%M:%S')
        except (ValueError, TypeError):
            formatted_date = processed_at
        
        # 認識信頼度
        confidence = round(result_data.get('confidence', 0), 1)
        
        # デバッグ情報（前処理画像など）
        debug_image_path = os.path.join(app.config['TEMP_FOLDER'], 'debug_preprocess.png')
        has_debug_image = os.path.exists(debug_image_path)
        
        # 元の画像のパス
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        has_original_image = os.path.exists(original_image_path)
        
        # テーブルセルの情報
        table_cells = result_data.get('table_cells', [])
        
        return render_template(
            'result.html', 
            filename=filename,
            original_filename=original_filename,
            formatted_date=formatted_date,
            result=result_data,
            structured_data=structured_data,
            confidence=confidence,
            has_debug_image=has_debug_image,
            has_original_image=has_original_image,
            table_cells=table_cells
        )
    
    except Exception as e:
        logger.error("Error displaying result: %s", str(e), exc_info=True)
        flash(f'結果ファイルの読み込み中にエラーが発生しました: {str(e)}')
        return redirect(url_for('dashboard'))

# デバッグ画像表示
@app.route('/debug_image/<image_type>')
def debug_image(image_type):
    if image_type == 'preprocessed':
        image_path = os.path.join(app.config['TEMP_FOLDER'], 'debug_preprocess.png')
    else:
        return "Invalid image type", 400
    
    if not os.path.exists(image_path):
        return "Image not found", 404
    
    return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))

# 元の画像表示
@app.route('/original_image/<filename>')
def original_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# データ管理画面
@app.route('/data')
@login_required
def data_management():
    # データベースからすべての不動産情報を取得
    properties = db.get_all_properties()
    return render_template('data.html', properties=properties)

# データ詳細表示・編集画面
@app.route('/data/<filename>', methods=['GET', 'POST'])
def data_detail(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash('指定されたファイルが見つかりません')
        return redirect(url_for('data_management'))
    
    if request.method == 'POST':
        try:
            # フォームからのデータを取得
            updated_data = {
                'property_number': request.form.get('property_number', ''),
                'address': request.form.get('address', ''),
                'lot_number': request.form.get('lot_number', ''),
                'area': request.form.get('area', ''),
                'owner_name': request.form.get('owner_name', ''),
                'owner_address': request.form.get('owner_address', ''),
                'rights': request.form.getlist('rights')  # 複数の権利情報を取得
            }
            
            # 既存のデータを読み込み
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 構造化データを更新
            data['structured_data'].update(updated_data)
            
            # 更新日時を記録
            data['updated_at'] = datetime.now().isoformat()
            data['updated_by'] = session.get('username', 'unknown')
            
            # 編集履歴を追加
            if 'edit_history' not in data:
                data['edit_history'] = []
            
            data['edit_history'].append({
                'timestamp': datetime.now().isoformat(),
                'user': session.get('username', 'unknown'),
                'changes': updated_data
            })
            
            # 更新したデータを保存
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info("Data updated for file: %s", filename)
            flash('データが正常に更新されました')
            return redirect(url_for('data_detail', filename=filename))
        
        except Exception as e:
            logger.error("Error updating data: %s", str(e), exc_info=True)
            flash(f'データの更新中にエラーが発生しました: {str(e)}')
            return redirect(url_for('data_detail', filename=filename))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 編集履歴の取得とフォーマット
        edit_history = []
        if 'edit_history' in data:
            for edit in data['edit_history']:
                try:
                    timestamp = datetime.fromisoformat(edit['timestamp']).strftime('%Y/%m/%d %H:%M:%S')
                except (ValueError, TypeError):
                    timestamp = edit['timestamp']
                
                edit_history.append({
                    'timestamp': timestamp,
                    'user': edit['user'],
                    'changes': edit['changes']
                })
        
        # 処理日時のフォーマット
        processed_at = data.get('processed_at', '')
        try:
            formatted_date = datetime.fromisoformat(processed_at).strftime('%Y年%m月%d日 %H:%M:%S')
        except (ValueError, TypeError):
            formatted_date = processed_at
        
        return render_template('data_detail.html', 
                              filename=filename, 
                              data=data, 
                              formatted_date=formatted_date,
                              edit_history=edit_history)
    
    except Exception as e:
        logger.error("Error loading data: %s", str(e), exc_info=True)
        flash(f'データの読み込み中にエラーが発生しました: {str(e)}')
        return redirect(url_for('data_management'))

# ファイルの削除
@app.route('/data/delete/<filename>', methods=['POST'])
def delete_data(filename):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash('指定されたファイルが見つかりません')
        return redirect(url_for('data_management'))
    
    try:
        os.remove(file_path)
        logger.info("File deleted: %s", filename)
        flash('ファイルが正常に削除されました')
    except Exception as e:
        logger.error("Error deleting file: %s", str(e))
        flash(f'ファイルの削除中にエラーが発生しました: {str(e)}')
    
    return redirect(url_for('data_management'))

# ファイルのエクスポート
@app.route('/data/export/<filename>/<format>')
def export_data(filename, format):
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash('指定されたファイルが見つかりません')
        return redirect(url_for('data_management'))
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ファイル名から拡張子を除去
        base_filename = os.path.splitext(filename)[0]
        
        if format == 'csv':
            # CSVにエクスポート
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{base_filename}.csv")
            
            # 構造化データをデータフレームに変換
            df_data = {
                '不動産番号': [data['structured_data'].get('property_number', '')],
                '所在': [data['structured_data'].get('address', '')],
                '地番': [data['structured_data'].get('lot_number', '')],
                '地積': [data['structured_data'].get('area', '')],
                '所有者名': [data['structured_data'].get('owner_name', '')],
                '所有者住所': [data['structured_data'].get('owner_address', '')],
                '権利情報': [', '.join(data['structured_data'].get('rights', []))]
            }
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            logger.info("Exported to CSV: %s", output_path)
            return send_from_directory(app.config['OUTPUT_FOLDER'], f"{base_filename}.csv", as_attachment=True)
            
        elif format == 'json':
            # JSONでダウンロード（既存のJSONファイル）
            return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)
            
        else:
            flash(f'未対応のエクスポート形式です: {format}')
            return redirect(url_for('data_detail', filename=filename))
            
    except Exception as e:
        logger.error("Error exporting data: %s", str(e), exc_info=True)
        flash(f'データのエクスポート中にエラーが発生しました: {str(e)}')
        return redirect(url_for('data_detail', filename=filename))
        
# ヘルパーメソッド：ファイルから信頼度を取得
def _get_confidence_from_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('confidence', 0)
    except:
        return 0

# レポート生成画面
@app.route('/reports')
@login_required
def reports():
    # レポートテンプレート一覧
    report_templates = [
        {
            'id': 'property_list',
            'name': '不動産一覧表',
            'description': '処理した全不動産の基本情報一覧'
        },
        {
            'id': 'owner_list',
            'name': '所有者一覧表',
            'description': '所有者ごとの不動産情報一覧'
        },
        {
            'id': 'survey_sheet',
            'name': '土地登記簿調査表',
            'description': '土地の登記簿情報をまとめた調査表'
        },
        {
            'id': 'rights_report',
            'name': '権利関係レポート',
            'description': '権利関係の詳細情報レポート'
        }
    ]
    
    # 出力形式一覧
    output_formats = [
        {'id': 'csv', 'name': 'CSV形式'},
        {'id': 'excel', 'name': 'Excel形式'},
        {'id': 'pdf', 'name': 'PDF形式'}
    ]
    
    return render_template('reports.html', 
                          report_templates=report_templates,
                          output_formats=output_formats)

# レポート生成API
@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    report_type = request.form.get('report_type')
    format_type = request.form.get('format', 'csv')
    
    if not report_type:
        return jsonify({'error': 'レポートタイプが指定されていません'}), 400
    
    try:
        # 全データの読み込み
        all_data = []
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith('.json')]
            
            for file in files:
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.append(data)
                except Exception as e:
                    logger.error("Error reading file %s for report: %s", file, str(e))
        
        # レポートタイプに応じたデータ処理
        if report_type == 'property_list':
            report_data = self._generate_property_list_report(all_data)
        elif report_type == 'owner_list':
            report_data = self._generate_owner_list_report(all_data)
        elif report_type == 'survey_sheet':
            report_data = self._generate_survey_sheet_report(all_data)
        elif report_type == 'rights_report':
            report_data = self._generate_rights_report(all_data)
        else:
            return jsonify({'error': f'未対応のレポートタイプです: {report_type}'}), 400
        
        # レポートの出力
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{report_type}_{timestamp}"
        
        if format_type == 'csv':
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{report_filename}.csv")
            report_data.to_csv(output_path, index=False, encoding='utf-8-sig')
            download_url = url_for('download_file', filename=f"{report_filename}.csv")
            
        elif format_type == 'excel':
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{report_filename}.xlsx")
            report_data.to_excel(output_path, index=False, engine='openpyxl')
            download_url = url_for('download_file', filename=f"{report_filename}.xlsx")
            
        elif format_type == 'pdf':
            # PDFレポートの生成は別途実装が必要
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{report_filename}.pdf")
            self._generate_pdf_report(report_data, output_path, report_type)
            download_url = url_for('download_file', filename=f"{report_filename}.pdf")
            
        else:
            return jsonify({'error': f'未対応の出力形式です: {format_type}'}), 400
        
        logger.info("Report generated: %s (%s)", report_type, format_type)
        return jsonify({
            'success': True,
            'download_url': download_url,
            'message': 'レポートが正常に生成されました'
        })
        
    except Exception as e:
        logger.error("Error generating report: %s", str(e), exc_info=True)
        return jsonify({'error': f'レポート生成中にエラーが発生しました: {str(e)}'}), 500

# 不動産一覧レポートの生成
def _generate_property_list_report(all_data):
    rows = []
    for data in all_data:
        structured_data = data.get('structured_data', {})
        rows.append({
            '不動産番号': structured_data.get('property_number', ''),
            '所在': structured_data.get('address', ''),
            '地番': structured_data.get('lot_number', ''),
            '地積': structured_data.get('area', ''),
            '所有者名': structured_data.get('owner_name', ''),
            '信頼度': data.get('confidence', 0)
        })
    
    return pd.DataFrame(rows)

# 所有者一覧レポートの生成
def _generate_owner_list_report(all_data):
    # 所有者ごとにデータをグループ化
    owners_data = {}
    
    for data in all_data:
        structured_data = data.get('structured_data', {})
        owner_name = structured_data.get('owner_name', '不明')
        
        if owner_name not in owners_data:
            owners_data[owner_name] = []
            
        owners_data[owner_name].append({
            '不動産番号': structured_data.get('property_number', ''),
            '所在': structured_data.get('address', ''),
            '地番': structured_data.get('lot_number', ''),
            '地積': structured_data.get('area', '')
        })
    
    # 所有者ごとのデータをフラット化
    rows = []
    for owner, properties in owners_data.items():
        for i, prop in enumerate(properties):
            row = {'所有者名': owner if i == 0 else '', '所有不動産数': len(properties) if i == 0 else ''}
            row.update(prop)
            rows.append(row)
    
    return pd.DataFrame(rows)

# 土地登記簿調査表レポートの生成
def _generate_survey_sheet_report(all_data):
    rows = []
    for data in all_data:
        structured_data = data.get('structured_data', {})
        property_number = structured_data.get('property_number', '')
        
        # 土地データのみフィルタリング
        if property_number and structured_data.get('area'):
            rows.append({
                '調査日': datetime.now().strftime("%Y/%m/%d"),
                '不動産番号': property_number,
                '所在': structured_data.get('address', ''),
                '地番': structured_data.get('lot_number', ''),
                '地目': '宅地',  # OCRで抽出できていない場合はデフォルト値
                '地積(㎡)': structured_data.get('area', ''),
                '所有者名': structured_data.get('owner_name', ''),
                '所有者住所': structured_data.get('owner_address', ''),
                '権利情報': ', '.join(structured_data.get('rights', [])),
                '備考': ''
            })
    
    return pd.DataFrame(rows)

# 権利関係レポートの生成
def _generate_rights_report(all_data):
    rows = []
    for data in all_data:
        structured_data = data.get('structured_data', {})
        rights = structured_data.get('rights', [])
        
        if rights:
            for right in rights:
                rows.append({
                    '不動産番号': structured_data.get('property_number', ''),
                    '所在': structured_data.get('address', ''),
                    '地番': structured_data.get('lot_number', ''),
                    '権利情報': right,
                    '所有者名': structured_data.get('owner_name', '')
                })
    
    return pd.DataFrame(rows)

# PDFレポートの生成
def _generate_pdf_report(df, output_path, report_type):
    try:
        # PDFレポート生成の実装
        # ここではシンプルなPDFテーブルを生成するが、
        # 実際にはより複雑なレイアウトが必要になる場合がある
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(output_path) as pdf:
            fig, ax = plt.subplots(figsize=(11, 8.5))  # A4サイズ
            ax.axis('tight')
            ax.axis('off')
            
            # レポートタイトル
            if report_type == 'property_list':
                title = '不動産一覧表'
            elif report_type == 'owner_list':
                title = '所有者一覧表'
            elif report_type == 'survey_sheet':
                title = '土地登記簿調査表'
            elif report_type == 'rights_report':
                title = '権利関係レポート'
            else:
                title = 'レポート'
            
            plt.suptitle(title, fontsize=16)
            plt.title(f'生成日: {datetime.now().strftime("%Y年%m月%d日")}', fontsize=10)
            
            # データテーブルの作成
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                loc='center',
                cellLoc='center',
                bbox=[0.1, 0.1, 0.8, 0.7]  # [left, bottom, width, height]
            )
            
            # テーブルスタイルの調整
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)
            
            # PDFに保存
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        return True
    except Exception as e:
        logger.error("Error generating PDF report: %s", str(e), exc_info=True)
        raise e

# ファイルダウンロード
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

# 設定画面
@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    if request.method == 'POST':
        try:
            # OCR設定の更新
            ocr_settings = {
                'dpi': int(request.form.get('dpi', 300)),
                'psm': int(request.form.get('psm', 6)),
                'oem': int(request.form.get('oem', 3)),
                'confidence_threshold': int(request.form.get('confidence_threshold', 70))
            }
            
            # ユーザー設定の更新
            user_settings = {
                'username': request.form.get('username', 'user'),
                'language': request.form.get('language', 'ja'),
                'theme': request.form.get('theme', 'light')
            }
            
            # システム設定の更新
            system_settings = {
                'auto_backup': request.form.get('auto_backup') == 'on',
                'backup_interval': int(request.form.get('backup_interval', 24)),
                'max_upload_size': int(request.form.get('max_upload_size', 16))
            }
            
            # 設定を併合
            combined_settings = {**ocr_settings, **user_settings, **system_settings}
            
            # 設定をファイルに保存
            with open('config.json', 'w', encoding='utf-8') as f:
                json.dump(combined_settings, f, ensure_ascii=False, indent=2)
            
            # OCRエンジンの設定を更新
            ocr_engine.config.update(ocr_settings)
            
            # アプリケーション設定を更新
            app.config['MAX_CONTENT_LENGTH'] = system_settings['max_upload_size'] * 1024 * 1024
            
            # セッションにユーザー名を保存
            session['username'] = user_settings['username']
            session['theme'] = user_settings['theme']
            
            logger.info("Settings updated")
            flash('設定が正常に更新されました')
            return redirect(url_for('settings'))
        
        except Exception as e:
            logger.error("Error updating settings: %s", str(e), exc_info=True)
            flash(f'設定の更新中にエラーが発生しました: {str(e)}')
            return redirect(url_for('settings'))
    
    # 現在の設定を読み込み
    current_settings = {
        'dpi': ocr_engine.config.get('dpi', 300),
        'psm': ocr_engine.config.get('psm', 6),
        'oem': ocr_engine.config.get('oem', 3),
        'confidence_threshold': ocr_engine.config.get('confidence_threshold', 70),
        'username': session.get('username', 'user'),
        'language': 'ja',
        'theme': session.get('theme', 'light'),
        'auto_backup': True,
        'backup_interval': 24,
        'max_upload_size': app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)
    }
    
    # 保存済みの設定がある場合は読み込み
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                saved_settings = json.load(f)
                current_settings.update(saved_settings)
        except Exception as e:
            logger.error("Error loading config.json: %s", str(e))
    
    # PSMオプションの定義
    psm_options = {
        0: 'オリエンテーションと単語の自動検出（OSD）のみ',
        1: '自動ページセグメンテーションとOSD',
        2: '自動ページセグメンテーション（OSDなし、OCRなし）',
        3: '完全自動ページセグメンテーション（OSDなし）',
        4: '可変サイズの単一列として仮定',
        5: '均一なテキストブロックとして仮定',
        6: '均一なテキストブロックとして仮定（単一の均一なブロック）',
        7: '画像を単一のテキスト行として扱う',
        8: '画像を単一の単語として扱う',
        9: '画像を円の中の単一の単語として扱う',
        10: '画像を単一の文字として扱う',
        11: '疎なテキスト。あらゆる方向と順序でできるだけ多くのテキストを見つける',
        12: '疎なテキストと強制的なOSD',
        13: '生のライン。画像を単一のテキスト行として扱う（ヒューリスティックなモデル無効）'
    }
    
    # OEMオプションの定義
    oem_options = {
        0: 'レガシーエンジンのみ',
        1: 'ニューラルネットワークLSTMエンジンのみ',
        2: 'レガシー + LSTMエンジン',
        3: 'デフォルト（利用可能な最適なエンジン）'
    }
    
    # 言語オプション
    language_options = {
        'ja': '日本語',
        'en': '英語'
    }
    
    # テーマオプション
    theme_options = {
        'light': 'ライトモード',
        'dark': 'ダークモード'
    }
    
    return render_template('settings.html', 
                          settings=current_settings,
                          psm_options=psm_options,
                          oem_options=oem_options,
                          language_options=language_options,
                          theme_options=theme_options)

# 対話型インターフェース画面
@app.route('/chat')
def chat():
    return render_template('chat.html')

# 対話型インターフェースAPI
@app.route('/api/chat', methods=['POST'])
def chat_api():
    query = request.json.get('query', '')
    
    if not query:
        return jsonify({'error': 'クエリが空です'}), 400
    
    try:
        # 全データの検索
        all_data = []
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if f.endswith('.json')]
            
            for file in files:
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        all_data.append(data)
                except Exception as e:
                    logger.error("Error reading file %s for chat: %s", file, str(e))
        
        # クエリに基づいて検索
        results = self._search_data(query, all_data)
        
        # レスポンスの生成
        response = self._generate_chat_response(query, results)
        
        logger.info("Chat query processed: %s", query)
        return jsonify({
            'query': query,
            'response': response,
            'results': results[:5]  # 上位5件のみ返す
        })
        
    except Exception as e:
        logger.error("Error processing chat query: %s", str(e), exc_info=True)
        return jsonify({'error': f'クエリの処理中にエラーが発生しました: {str(e)}'}), 500

# テキスト検索機能
def _search_data(query, all_data):
    results = []
    
    # クエリからキーワードを抽出
    keywords = query.lower().split()
    
    # 不動産番号の検索パターン
    property_pattern = r'\d{4}-\d{4}-\d{4}'
    property_matches = re.findall(property_pattern, query)
    
    for data in all_data:
        score = 0
        structured_data = data.get('structured_data', {})
        
        # 不動産番号の完全一致（最も優先度が高い）
        if property_matches and structured_data.get('property_number') in property_matches:
            score += 100
        
        # 各フィールドとキーワードのマッチング
        for field in ['property_number', 'address', 'lot_number', 'area', 'owner_name', 'owner_address']:
            field_value = str(structured_data.get(field, '')).lower()
            
            # キーワードとのマッチ
            for keyword in keywords:
                if keyword in field_value:
                    # フィールドの重要度に応じたスコア付け
                    if field in ['property_number', 'owner_name']:
                        score += 10
                    elif field in ['address', 'lot_number']:
                        score += 5
                    else:
                        score += 2
        
        # 権利情報の検索
        rights = structured_data.get('rights', [])
        rights_text = ' '.join(rights).lower()
        
        for keyword in keywords:
            if keyword in rights_text:
                score += 3
        
        # スコアが0より大きい場合は結果に追加
        if score > 0:
            results.append({
                'file_name': data.get('file_name', 'Unknown'),
                'property_number': structured_data.get('property_number', 'N/A'),
                'address': structured_data.get('address', 'N/A'),
                'owner_name': structured_data.get('owner_name', 'N/A'),
                'score': score,
                'result_file': os.path.basename(data.get('output_path', ''))
            })
    
    # スコアでソート
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results

# チャットレスポンスの生成
def _generate_chat_response(query, results):
    # 結果が見つからない場合
    if not results:
        return "申し訳ありません。該当する登記簿データが見つかりませんでした。別のキーワードで検索するか、より具体的な情報を入力してください。"
    
    # 質問の種類を判断
    if '誰' in query or '所有者' in query or '名義人' in query:
        # 所有者に関する質問
        if len(results) == 1:
            return f"「{results[0]['address']} {results[0]['property_number']}」の所有者は「{results[0]['owner_name']}」です。"
        else:
            owners = [f"「{r['address']}」の所有者は「{r['owner_name']}」" for r in results[:3]]
            return f"検索結果: {', '.join(owners)}です。もっと詳しく知りたい場合は、不動産番号や住所をより具体的に指定してください。"
    
    elif '住所' in query or 'どこ' in query or '所在' in query:
        # 住所に関する質問
        if len(results) == 1:
            return f"不動産番号「{results[0]['property_number']}」の所在地は「{results[0]['address']}」です。"
        else:
            addresses = [f"「{r['property_number']}」の所在地は「{r['address']}」" for r in results[:3]]
            return f"検索結果: {', '.join(addresses)}です。"
    
    elif '土地' in query or '地番' in query:
        # 土地に関する質問
        if len(results) == 1:
            structured_data = self._get_structured_data_from_result(results[0])
            if structured_data:
                return f"「{structured_data.get('address', 'N/A')}」の地番は「{structured_data.get('lot_number', 'N/A')}」、地積は「{structured_data.get('area', 'N/A')}」です。"
        
        return f"検索条件に一致する土地情報が{len(results)}件見つかりました。詳細を確認するには、より具体的な情報を入力してください。"
    
    else:
        # 一般的な検索結果
        if len(results) == 1:
            return f"検索結果: 不動産番号「{results[0]['property_number']}」、所在地「{results[0]['address']}」、所有者「{results[0]['owner_name']}」のデータが見つかりました。"
        else:
            return f"検索条件に一致するデータが{len(results)}件見つかりました。主な結果: {', '.join([r['property_number'] for r in results[:3]])}。もっと詳しく知りたい場合は、より具体的なキーワードで検索してください。"

# 結果からstructured_dataを取得
def _get_structured_data_from_result(result):
    result_file = result.get('result_file')
    if not result_file:
        return None
    
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], result_file)
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('structured_data', {})
    except:
        return None

# タイムライン表示
@app.route('/timeline/<property_id>')
def timeline(property_id):
    # データベースから不動産情報を取得
    property_data = None
    properties = db.get_all_properties()
    for prop in properties:
        if str(prop['id']) == property_id:
            property_data = prop
            break
    
    if not property_data:
        flash('指定された不動産情報が見つかりません', 'danger')
        return redirect(url_for('data_management'))
    
    # 所有者変更履歴を取得
    timeline_data = db.get_property_timeline(property_id)
    
    return render_template('timeline.html', property=property_data, timeline=timeline_data)

# 差分検出・比較表示
@app.route('/diff/<file1>/<file2>')
def diff_view(file1, file2):
    file1_path = os.path.join(app.config['OUTPUT_FOLDER'], file1)
    file2_path = os.path.join(app.config['OUTPUT_FOLDER'], file2)
    
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        flash('比較対象のファイルが見つかりません')
        return redirect(url_for('data_management'))
    
    try:
        # 2つのJSONファイルを読み込み
        with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        
        # ファイル情報
        file_info = {
            'file1': {
                'filename': data1.get('file_name', 'Unknown'),
                'processed_at': data1.get('processed_at', '')
            },
            'file2': {
                'filename': data2.get('file_name', 'Unknown'),
                'processed_at': data2.get('processed_at', '')
            }
        }
        
        # 日付フォーマット
        for file_key in ['file1', 'file2']:
            try:
                dt = datetime.fromisoformat(file_info[file_key]['processed_at'])
                file_info[file_key]['formatted_date'] = dt.strftime('%Y年%m月%d日 %H:%M:%S')
            except (ValueError, TypeError):
                file_info[file_key]['formatted_date'] = file_info[file_key]['processed_at']
        
        # 構造化データの差分を計算
        structured_data1 = data1.get('structured_data', {})
        structured_data2 = data2.get('structured_data', {})
        
        # 差分情報を格納する辞書
        diff_data = {}
        
        # 全てのキーを取得
        all_keys = set(structured_data1.keys()) | set(structured_data2.keys())
        
        for key in all_keys:
            value1 = structured_data1.get(key, None)
            value2 = structured_data2.get(key, None)
            
            # 値の型に応じた比較
            if isinstance(value1, list) and isinstance(value2, list):
                # リストの場合は要素ごとに比較
                if sorted(value1) != sorted(value2):
                    diff_data[key] = {
                        'value1': value1,
                        'value2': value2,
                        'status': 'changed'
                    }
                else:
                    diff_data[key] = {
                        'value1': value1,
                        'value2': value2,
                        'status': 'unchanged'
                    }
            else:
                # 一般的な値の比較
                if value1 != value2:
                    status = 'changed'
                    if value1 is None:
                        status = 'added'
                    elif value2 is None:
                        status = 'removed'
                    
                    diff_data[key] = {
                        'value1': value1,
                        'value2': value2,
                        'status': status
                    }
                else:
                    diff_data[key] = {
                        'value1': value1,
                        'value2': value2,
                        'status': 'unchanged'
                    }
        
        # 変更の要約
        summary = {
            'total_fields': len(all_keys),
            'changed_fields': len([k for k, v in diff_data.items() if v['status'] != 'unchanged']),
            'added_fields': len([k for k, v in diff_data.items() if v['status'] == 'added']),
            'removed_fields': len([k for k, v in diff_data.items() if v['status'] == 'removed']),
            'modified_fields': len([k for k, v in diff_data.items() if v['status'] == 'changed'])
        }
        
        return render_template('diff.html',
                              file_info=file_info,
                              diff_data=diff_data,
                              summary=summary)
    
    except Exception as e:
        logger.error("Error generating diff view: %s", str(e), exc_info=True)
        flash(f'差分表示中にエラーが発生しました: {str(e)}')
        return redirect(url_for('data_management'))

# エラーハンドリング
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error='404 Not Found', message='ページが見つかりません'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error='500 Internal Server Error', message='サーバー内部でエラーが発生しました'), 500

# ログイン画面
@app.route('/login', methods=['GET', 'POST'])
def login():
    # ログイン済みの場合はダッシュボードへリダイレクト
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember')
        
        # ログイン認証
        user = auth.login(username, password)
        if user:
            # セッションにユーザー情報を保存
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']
            
            # Remember Me機能
            if remember:
                # セッションの有効期限を延長（デフォルトは30日）
                session.permanent = True
                app.permanent_session_lifetime = timedelta(days=30)
            
            flash('ログインしました', 'success')
            
            # リダイレクト先の取得
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('ユーザー名またはパスワードが正しくありません', 'danger')
    
    return render_template('login.html')

# ログアウト
@app.route('/logout')
def logout():
    # セッションからユーザー情報を削除
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('user_role', None)
    
    flash('ログアウトしました', 'success')
    return redirect(url_for('login'))

# ユーザー管理画面（管理者専用）
@app.route('/users')
@admin_required
def user_management():
    # すべてのユーザーを取得
    users = []
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users ORDER BY username')
        rows = cursor.fetchall()
        
        for row in rows:
            users.append(dict(row))
    
    return render_template('user_management.html', users=users)

# 新規ユーザー追加（管理者専用）
@app.route('/users/add', methods=['POST'])
@admin_required
def add_user():
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')
    role = request.form.get('role')
    
    if not username or not password:
        flash('ユーザー名とパスワードは必須です', 'danger')
        return redirect(url_for('user_management'))
    
    # ユーザー登録
    if auth.register_user(username, password, email, role):
        flash(f'ユーザー「{username}」を追加しました', 'success')
    else:
        flash(f'ユーザー「{username}」の追加に失敗しました', 'danger')
    
    return redirect(url_for('user_management'))

# ユーザー編集（管理者専用）
@app.route('/users/edit/<int:user_id>', methods=['POST'])
@admin_required
def edit_user(user_id):
    email = request.form.get('email')
    role = request.form.get('role')
    new_password = request.form.get('new_password')
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # パスワード変更がある場合
            if new_password:
                password_hash = auth._hash_password(new_password)
                cursor.execute('UPDATE users SET email = ?, role = ?, password_hash = ? WHERE id = ?',
                             (email, role, password_hash, user_id))
            else:
                cursor.execute('UPDATE users SET email = ?, role = ? WHERE id = ?',
                             (email, role, user_id))
            
            conn.commit()
            flash('ユーザー情報を更新しました', 'success')
    except Exception as e:
        logger.error("Failed to update user: %s", str(e), exc_info=True)
        flash('ユーザー情報の更新に失敗しました', 'danger')
    
    return redirect(url_for('user_management'))

# ユーザー削除（管理者専用）
@app.route('/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def delete_user(user_id):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 削除前にユーザー名を取得
            cursor.execute('SELECT username FROM users WHERE id = ?', (user_id,))
            username = cursor.fetchone()[0]
            
            # adminユーザーは削除不可
            if username == 'admin':
                flash('管理者ユーザーは削除できません', 'danger')
                return redirect(url_for('user_management'))
            
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            
            flash(f'ユーザー「{username}」を削除しました', 'success')
    except Exception as e:
        logger.error("Failed to delete user: %s", str(e), exc_info=True)
        flash('ユーザーの削除に失敗しました', 'danger')
    
    return redirect(url_for('user_management'))

# パスワード変更（一般ユーザー用）
@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    current_password = request.form.get('current_password')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    if not current_password or not new_password or not confirm_password:
        flash('すべての項目を入力してください', 'danger')
        return redirect(url_for('settings'))
    
    if new_password != confirm_password:
        flash('新しいパスワードと確認用パスワードが一致しません', 'danger')
        return redirect(url_for('settings'))
    
    user_id = session.get('user_id')
    if auth.change_password(user_id, current_password, new_password):
        flash('パスワードを変更しました', 'success')
    else:
        flash('現在のパスワードが正しくありません', 'danger')
    
    return redirect(url_for('settings'))

# アプリケーション起動
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
