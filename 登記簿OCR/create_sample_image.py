"""
サンプル登記簿画像生成スクリプト

このスクリプトはテスト用のサンプル登記簿画像を生成します。
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse

def create_sample_registry_image(output_path, input_file=None, font_path=None):
    """
    登記簿謄本のサンプル画像を生成します。
    A4サイズ、300DPIの画像を作成し、テキストとテーブル枠を追加します。
    スキャンしたような外観を再現するためにノイズと傾きを追加します。
    
    Args:
        output_path (str): 出力する画像ファイルのパス
        input_file (str): サンプルテキストを含むファイルのパス（指定しない場合はデフォルトのサンプルファイルを使用）
        font_path (str): 使用するフォントのパス（指定しない場合はシステムに応じたデフォルトフォントを使用）
    """
    # デフォルトのフォントパスを設定（OSに応じて）
    if not font_path:
        if sys.platform == 'darwin':  # macOS
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
            if not os.path.exists(font_path):
                font_path = '/Library/Fonts/Arial Unicode.ttf'
        elif sys.platform == 'win32':  # Windows
            font_path = 'C:\\Windows\\Fonts\\msgothic.ttc'
        else:  # Linux その他
            font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    
    # フォントファイルが存在するか確認
    if not os.path.exists(font_path):
        print(f"警告: フォントファイル '{font_path}' が見つかりません。デフォルトフォントを使用します。")
        font_path = None
    
    # サンプルテキストを読み込む
    sample_text = ""
    if input_file and os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            sample_text = f.read()
    elif os.path.exists('sample_data/sample_registry.txt'):
        with open('sample_data/sample_registry.txt', 'r', encoding='utf-8') as f:
            sample_text = f.read()
    else:
        # デフォルトのサンプルテキスト
        sample_text = """不動産番号：2023-1234-5678
所在：東京都千代田区
地番：1-1
地積：1,000.25平方メートル

甲区（所有権に関する事項）
順位番号：1
登記の目的：所有権移転
受付年月日：令和5年6月1日
登記原因：売買
権利者：山田太郎
住所：東京都新宿区西新宿2-8-1

乙区（所有権以外の権利に関する事項）
順位番号：1
登記の目的：抵当権設定
受付年月日：令和5年6月15日
登記原因：令和5年6月10日金銭消費貸借契約
債権額：金100,000,000円
利息：年3%
遅延損害金：年14%
債務者：山田太郎
抵当権者：東京銀行 代表者 鈴木一郎"""
    
    # A4サイズを300DPIで作成（A4 = 210mm x 297mm）
    width = int(210 * 300 / 25.4)  # mmをピクセルに変換
    height = int(297 * 300 / 25.4)
    
    # 白い背景の画像を作成
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        # タイトル用フォント（大きめ）
        title_font = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
        # 本文用フォント
        body_font = ImageFont.truetype(font_path, 28) if font_path else ImageFont.load_default()
    except Exception as e:
        print(f"フォント読み込みエラー: {e}")
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
    
    # タイトルを描画
    title = "登記簿謄本"
    title_width = draw.textlength(title, font=title_font)
    draw.text(((width - title_width) / 2, 50), title, font=title_font, fill='black')
    
    # 本文テキストを描画
    lines = sample_text.strip().split('\n')
    y_position = 120
    for line in lines:
        draw.text((50, y_position), line, font=body_font, fill='black')
        y_position += 40
        
        # 甲区、乙区の区切り線を描画
        if "甲区" in line or "乙区" in line:
            draw.line([(50, y_position), (width - 50, y_position)], fill='black', width=2)
            y_position += 10
    
    # テーブル枠を描画
    draw.rectangle([(40, 110), (width - 40, min(y_position + 50, height - 40))], outline='black', width=2)
    
    # スキャンしたような見た目にするためにわずかに回転
    angle = np.random.uniform(-0.5, 0.5)  # -0.5度から0.5度のランダムな角度
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
    
    # ノイズを追加
    img_array = np.array(image)
    noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    noisy_image = Image.fromarray(noisy_img_array)
    
    # 画像を保存
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        noisy_image.save(output_path)
        print(f"サンプル登記簿画像を作成しました: {output_path}")
        return True
    except Exception as e:
        print(f"画像保存エラー: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='登記簿謄本のサンプル画像を生成します')
    parser.add_argument('-o', '--output', default='sample_data/sample_registry.jpg', 
                        help='出力する画像ファイルのパス')
    parser.add_argument('-i', '--input', default=None, 
                        help='サンプルテキストを含むファイルのパス')
    parser.add_argument('-f', '--font', default=None, 
                        help='使用するフォントのパス')
    
    args = parser.parse_args()
    
    create_sample_registry_image(args.output, args.input, args.font) 