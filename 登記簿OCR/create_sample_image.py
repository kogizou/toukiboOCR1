"""
サンプル登記簿画像生成スクリプト

このスクリプトはテスト用のサンプル登記簿画像を生成します。
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_sample_registry_image(output_path, font_path=None):
    """
    サンプルの登記簿画像を生成する
    
    Args:
        output_path (str): 出力する画像ファイルのパス
        font_path (str, optional): フォントファイルのパス
    """
    # デフォルトのフォントパス
    if font_path is None:
        if sys.platform == 'darwin':  # Mac
            font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'
        elif sys.platform == 'win32':  # Windows
            font_path = 'C:\\Windows\\Fonts\\msgothic.ttc'
        else:  # Linux
            font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    
    # フォントファイルが存在しない場合はデフォルトフォントを使用
    if not os.path.exists(font_path):
        print(f"Font file not found: {font_path}")
        print("Using default font")
        font_path = None
    
    # サンプルテキストの読み込み
    try:
        with open('sample_data/sample_registry.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        text = """不動産番号：2023-1234-5678

所在　東京都千代田区丸の内一丁目
地番　1番1
地積　1,000.25平方メートル

権利部（甲区）

順位番号　1
登記の目的　所有権移転
受付年月日　令和5年6月1日
登記原因　売買
権利者　氏名　山田太郎
　　　　住所　東京都新宿区西新宿三丁目2番3号

権利部（乙区）

順位番号　1
登記の目的　抵当権設定
受付年月日　令和5年6月15日
登記原因　令和5年6月10日金銭消費貸借契約
債権額　金100,000,000円
利息　年3パーセント
遅延損害金　年14パーセント
債務者　山田太郎
抵当権者　株式会社東京銀行
　　　　　代表取締役　鈴木一郎"""

    # 画像の作成（A4サイズ、300DPI）
    width = int(8.27 * 300)  # A4幅 (8.27インチ)
    height = int(11.7 * 300)  # A4高さ (11.7インチ)
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    
    try:
        # フォントの設定
        title_font_size = 36
        body_font_size = 24
        
        if font_path:
            title_font = ImageFont.truetype(font_path, title_font_size)
            body_font = ImageFont.truetype(font_path, body_font_size)
        else:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
        
        # 描画オブジェクトの作成
        draw = ImageDraw.Draw(image)
        
        # タイトル
        draw.text((300, 100), "登記簿謄本", font=title_font, fill=(0, 0, 0))
        
        # 本文
        lines = text.split('\n')
        y_position = 200
        line_height = body_font_size + 10
        
        for line in lines:
            draw.text((100, y_position), line, font=body_font, fill=(0, 0, 0))
            y_position += line_height
        
        # テーブル境界線の描画
        draw.line([(50, 180), (width-50, 180)], fill=(0, 0, 0), width=2)  # 上部境界線
        draw.line([(50, 180), (50, y_position + 50)], fill=(0, 0, 0), width=2)  # 左境界線
        draw.line([(width-50, 180), (width-50, y_position + 50)], fill=(0, 0, 0), width=2)  # 右境界線
        draw.line([(50, y_position + 50), (width-50, y_position + 50)], fill=(0, 0, 0), width=2)  # 下部境界線
        
        # 権利部の区切り線
        for i, line in enumerate(lines):
            if "権利部" in line:
                line_y = 200 + i * line_height
                draw.line([(50, line_y - 10), (width-50, line_y - 10)], fill=(0, 0, 0), width=1)
        
        # 少しノイズを加える（より実際の文書のような見た目にするため）
        noise = np.random.normal(0, 5, (height, width, 3)).astype(np.uint8)
        noisy_image = np.array(image) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_image)
        
        # 少し回転させる（リアルなスキャン文書のように）
        angle = np.random.uniform(-0.5, 0.5)  # 0.5度以内の回転
        image = image.rotate(angle, resample=Image.BICUBIC, expand=False)
        
        # 画像の保存
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, dpi=(300, 300))
        print(f"Sample registry image created: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating sample image: {str(e)}")
        return False

if __name__ == "__main__":
    # サンプルデータディレクトリの作成
    os.makedirs('sample_data', exist_ok=True)
    
    # サンプル画像の生成
    output_path = 'sample_data/sample_registry.jpg'
    success = create_sample_registry_image(output_path)
    
    if success:
        print(f"サンプル登記簿画像を生成しました: {output_path}")
    else:
        print("サンプル画像の生成に失敗しました") 