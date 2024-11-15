import cv2
import base64
import streamlit as st
from openai import OpenAI, OpenAIError
import os

#import config # ローカル実行
import PyPDF2
import json
import pandas as pd
import io  # メモリ上にファイルを保存するために必要
from PIL import Image
import numpy as np
file_type = ""
system_prompt = """
# 役割:
- あなたは提供された製品変更通知（PCN）や${file_type}の内容を、以下のキーに従って構造化し、整理されたjsonデータを出力します。

# 制約:
- 各キーには該当する情報を正確に埋めてください。
- {file_type}にその情報が含まれていない場合は「N/A」と記載します。
- 提供された情報を以下のキーの順に従って整理してください。
- 無関係な情報は含めず、特に関連のあるデータに集中してください。
- これらの制約を守らなければ、ユーザーに大きな不利益をもたらします。

# キー構造:
1. FPCN番号:  
   - 説明: 各文書に付与された識別番号を記載。
   - フォーマット: FPCNXXXXX
   - FPCNから始まり、後ろには数字が5桁続きます。
   - 例:FPCN12345

2. 発行日:  
   - 説明: 文書が発行された日付を記載。
   - フォーマット: YYYY/MM/DD（例: 2023/12/01）

3. 部品番号(Part Number):  
   - 説明: 変更が適用される部品番号を記載。
   - 認定試験用ビーグル(Qualification Vehicle)と必ず対応するペアがあります。
   - 認定試験用ビーグル(Qualification Vehicle)と部品番号(Part Number)は1:nの関係になります。

4. 認定試験用ビークル(Qualification Vehicle)
   - 説明: 品質や性能を確認するためのテスト用部品の番号を記載。
   - 部品番号(Part Number)と必ず対応するペアがあります。
   - 認定試験用ビーグル(Qualification Vehicle)と部品番号(Part Number)は1:nの関係になります。

# 出力フォーマット
余計なコメントは言わず、jsonデータのみ出力してください。
jsonで以下の形式で必ず整理してください。
```で囲んだり、頭にjsonとつけてはいけません。
{{
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCNXXXXX", "2024/01/20", "XXX", "YYY"],
    ...
  ]
}}
"""

pdf_review_prompt = """
# 役割
- あなたはJSONデータと元データを使用してチェック観点に従って、レビューを行い正しい場合はJSONデータを出力し、正しくない場合はJSONデータを修正して出力します。

# 前提知識
- JSONデータは提供されたプロンプトをもとに元データから作成されたデータです。
- JSONデータの構造は以下です。
- "columns"をKEY項目に格納されている配列はラベル項目です。固定で["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"]が格納されています。
- "data"をKEY項目に格納されている配列は元データから抽出したデータで、変更が適用される部品番号単位で配列が格納されています。
- "data"をKEY項目に格納されている配列に格納されている部品番号単位毎の文字列の配列は、順に"FPCN番号","発行日","部品番号","認定試験用ビークル"に該当する値が格納されています。
- "FPCN番号","発行日","部品番号","認定試験用ビークル"に対応する値がどのように格納されているかは以下を参照してください。
  - "FPCN番号"
    - 説明: 元データの識別番号を抽出して格納しています。※部品番号とは異なるので注意してください。
    - サンプルデータ: "FPCN12345"
    - 特徴: "FPCN"から始まり、後ろに数字が5桁あります。
  - "発行日"
    - 説明: 元データの発行された日付を抽出して格納しています。※JSONデータには日付フォーマットされて格納されています。
    - 日付フォーマット形式: YYYY/MM/DD
    - サンプルデータ: "2023/12/01"
    - 特徴: 元データは英語で日付が記載されています。JSONデータは日付フォーマットされて格納されています。
  - "部品番号"
    - 説明: 元データの変更が適用される部品番号(Part Number)を抽出して格納しています。※FPCN番号とは異なるので注意してください。
    - サンプルデータ: "DF6A6.8FUT1G"
    - 特徴1: 英数字と記号が混在した文字列。※記号は必ずしも含んではいません。
    - 特徴2: 元データに部品番号(Part Number)に対応する認定試験用ビーグル(Qualification Vehicle)があります。
    - 特徴3: 部品番号(Part Number)と対応する認定試験用ビーグル(Qualification Vehicle)は同一の場合があります。
  - "認定試験用ビークル"
    - 説明: 元データの認定試験用ビークル(Qualification Vehicle)を抽出して格納しています。
    - サンプルデータ: "MMBZ47VALT1G"
    - 特徴1: 英数字が混在した文字列。
    - 特徴2: 配列に格納した元データの部品番号(Part Number)に対応する認定試験用ビーグル(Qualification Vehicle)を格納しています。
    - 特徴3: 認定試験用ビーグル(Qualification Vehicle)と部品番号(Part Number)は1:nの関係になります。
    - 特徴4: 部品番号(Part Number)と対応する認定試験用ビーグル(Qualification Vehicle)は同一の場合があります。
- 以下はJSONデータの例になります。
{
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCN12345", "2024/01/20", "X1X.X3", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X2X.X4", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X3X.X5", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "Y1Y2Y3", "Y1Y2Y3"],
  ]
}

# 制約:
- 必ず前提知識をベースにして、以下のレビュー項目に従ってレビューを行ってください。
- JSONデータは提供されたプロンプトをもとに元データから作成されたデータということを念頭に置いてレビューを行ってください。。
- 修正を行う場合は無関係な情報は含めず、特に関連のあるデータに集中してください。
- 必ず以下の出力フォーマットに従って出力してください。
- これらの制約を守らなければ、ユーザーに大きな不利益をもたらします。

# レビュー項目
1. 出力データは足りているか※最も重要
   - 以下の手順でレビューしてください。
   ※Qualification Vehicleは部品番号ではないので注意してください。
   1. 元データを確認し、変更が適用される部品番号(Part Number)を全て抽出してください。
   2. 本当に全て抽出できているか再度確認してください。
   3-1. 抽出した部品番号(Part Number)を順番に取り出します。
   3-2. JSONデータの"data"をKEY項目に格納されている配列の要素を順次確認して"部品番号"に該当する値と抽出した部品番号(Part Number)が同じか判定します。
   3-3. 抽出した部品番号(Part Number)と同じ値が見つからなかった場合、その部品番号(Part Number)の出力データは足りていません。
   4-1. 不足している部品番号(Part Number)があった場合、本当にその部品番号(Part Number)が元データに存在するか確認してください。
   4-2. 存在する場合、その部品番号(Part Number)の出力データは足りていません。
2. 項目の欠落がないか
   - 元データに基づき、FPCN番号, 発行日, 部品番号, 認定試験用ビークルの項目が抜け漏れがなく全て出力されているか確認する。
   - FPCN番号, 発行日はフォーマットで変換されていることを考慮して確認してください。
3. 項目のフォーマットは指定通りか
   - FPCN番号: 元データのFPCN番号が正しく出力されているか。
   - 発行日: 元データの発行日が日付フォーマット形式で出力されているか。
   - 部品番号:元データの部品番号が正しく出力されているか。
   - 認定試験用ビークル:元データの認定試験用ビークルが正しく出力されているか。
4. 余計な情報が出力されていないか
   - 元データに存在しない余分な情報がJSONに出力されていないか確認する
5. 部品番号と認定試験用ビークルのペアの正確性
   - 元データに基づき、JSONデータの各データの部品番号と認定試験用ビークルのペアが一致しているかを確認する。
   - 認定試験用ビーグルと部品番号は1:nの関係であることに注意してください。

# 出力フォーマット
- "results"にはレビュー項目毎の結果を出力してください。
- "message"にはレビュー項目毎の修正内容を出力してください。
- "data"部分は提供されたJSONデータもしくは修正を行ったものを出力してください。
- 余計なコメントは言わず、jsonデータのみ出力してください。
- jsonで以下の形式で必ず整理してください。
- ```で囲んだり、頭にjsonとつけてはいけません。
{
  "results": [
   "{レビュー項目1.の結果を"true" or "false"で出力してください}",
   "{レビュー項目2.の結果を"true" or "false"で出力してください}",
   "{レビュー項目3.の結果を"true" or "false"で出力してください}",
   "{レビュー項目4.の結果を"true" or "false"で出力してください}",
   "{レビュー項目5.の結果を"true" or "false"で出力してください}"
  ],
  "messages: [
   "{レビュー項目1.の修正内容を「部品番号[{不足している部品番号}]が出力されていませんでした。部品番号[{不足している部品番号}]を追加しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目2.の修正内容を「項目[{欠落している項目}]が欠落しています。項目[{欠落している項目}]に[{追加した値}]を追加しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目3.の修正内容を「項目[{指定フォーマットになっていない項目}]の[{指定フォーマットでない値}]がフォーマットが誤っています。[{修正した値}]に修正しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目4.の修正内容を「{余分な情報の内容}は存在しない情報です。削除しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目5.の修正内容を「[{部品番号}]と[{認定試験用ビーグル}]は正しいペアではありません。[{修正した部品番号}]と[{修正した認定試験用ビーグル}]に修正しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}"
  ],
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCN12345", "2024/01/20", "X1X.X3", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X2X.X4", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X3X.X5", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "Y1Y2Y3", "Y1Y2Y3"],
    ...
  ]
}
"""

reminder_prompt = """
# 制約:
- 必ず前提知識をベースにして、レビュー項目に従ってレビューを行ってください。
- JSONデータは提供されたプロンプトをもとに元データから作成されたデータということを念頭に置いてレビューを行ってください。。
- 修正を行う場合は無関係な情報は含めず、特に関連のあるデータに集中してください。
- 必ず出力フォーマットに従って出力してください。
- レビューした結果、修正が必要な場合は必ずJSONデータを修正して出力してください。
- 必ず修正した内容は全て出力してください。
- 指摘する値と修正している値が同じ場合があります。これは重大な誤りです。気をつけてください。
- これらの制約を守らなければ、ユーザーに大きな不利益をもたらします。
"""

question_prompt = """
# 役割
- あなたは以下の前提知識、提供されたJSONデータと元データとプロンプトを用いてユーザーの質問に対して正確な回答を答えを出力します。

# 前提知識
- JSONデータは提供されたプロンプトをもとに元データから作成されたデータです。
- JSONデータの構造は以下です。
- "columns"をKEY項目に格納されている配列はラベル項目です。固定で["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"]が格納されています。
- "data"をKEY項目に格納されている配列は元データから抽出したデータで、変更が適用される部品番号単位で配列が格納されています。
- "data"をKEY項目に格納されている配列に格納されている部品番号単位毎の文字列の配列は、順に"FPCN番号","発行日","部品番号","認定試験用ビークル"に該当する値が格納されています。
- "FPCN番号","発行日","部品番号","認定試験用ビークル"に対応する値がどのように格納されているかは以下を参照してください。
  - "FPCN番号"
    - 説明: 元データの識別番号を抽出して格納しています。※部品番号とは異なるので注意してください。
    - サンプルデータ: "FPCN12345"
    - 特徴: "FPCN"から始まり、後ろに数字が5桁あります。
  - "発行日"
    - 説明: 元データの発行された日付を抽出して格納しています。※JSONデータには日付フォーマットされて格納されています。
    - 日付フォーマット形式: YYYY/MM/DD
    - サンプルデータ: "2023/12/01"
    - 特徴: 元データは英語で日付が記載されています。JSONデータは日付フォーマットされて格納されています。
  - "部品番号"
    - 説明: 元データの変更が適用される部品番号(Part Number)を抽出して格納しています。※FPCN番号とは異なるので注意してください。
    - サンプルデータ: "DF6A6.8FUT1G"
    - 特徴1: 英数字と記号が混在した文字列。※記号は必ずしも含んではいません。
    - 特徴2: 元データに部品番号(Part Number)に対応する認定試験用ビーグル(Qualification Vehicle)があります。
    - 特徴3: 部品番号(Part Number)と対応する認定試験用ビーグル(Qualification Vehicle)は同一の場合があります。
  - "認定試験用ビークル"
    - 説明: 元データの認定試験用ビークル(Qualification Vehicle)を抽出して格納しています。
    - サンプルデータ: "MMBZ47VALT1G"
    - 特徴1: 英数字が混在した文字列。
    - 特徴2: 配列に格納した元データの部品番号(Part Number)に対応する認定試験用ビーグル(Qualification Vehicle)を格納しています。
    - 特徴3: 認定試験用ビーグル(Qualification Vehicle)と部品番号(Part Number)は1:nの関係になります。
    - 特徴4: 部品番号(Part Number)と対応する認定試験用ビーグル(Qualification Vehicle)は同一の場合があります。
- 以下はJSONデータの例になります。
{
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCN12345", "2024/01/20", "X1X.X3", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X2X.X4", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X3X.X5", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "Y1Y2Y3", "Y1Y2Y3"],
  ]
}

# 制約:
- 必ず前提知識ベースに回答を考えてください。
- 無関係な情報は含めず、特に関連のあるデータに集中してください。
- 回答は日本語の文章で出力してください。
- 回答が困難な場合は「提供された情報ではわかりません。」と回答してください。
- これらの制約を守らなければ、ユーザーに大きな不利益をもたらします。
"""

image_review_prompt = """
# 役割
- あなたはJSONデータと画像データを使用してチェック観点に従って、レビューを行い正しい場合はJSONデータを出力し、正しくない場合はJSONデータを修正して出力します。

# 前提知識
- JSONデータは提供されたプロンプトをもとに画像データから作成されたデータです。
- JSONデータの構造は以下です。
- "columns"をKEY項目に格納されている配列はラベル項目です。固定で["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"]が格納されています。
- "data"をKEY項目に格納されている配列は画像データから抽出したデータで、変更が適用される部品番号単位で配列が格納されています。
- "data"をKEY項目に格納されている配列に格納されている部品番号単位毎の文字列の配列は、順に"FPCN番号","発行日","部品番号","認定試験用ビークル"に該当する値が格納されています。
- "FPCN番号","発行日","部品番号","認定試験用ビークル"に対応する値がどのように格納されているかは以下を参照してください。
  - "FPCN番号"
    - 説明: 画像データの識別番号を抽出して格納しています。※部品番号とは異なるので注意してください。
    - サンプルデータ: "FPCN12345"
    - 特徴: "FPCN"から始まり、後ろに数字が5桁あります。
  - "発行日"
    - 説明: 画像データの発行された日付を抽出して格納しています。※JSONデータには日付フォーマットされて格納されています。
    - 日付フォーマット形式: YYYY/MM/DD
    - サンプルデータ: "2023/12/01"
    - 特徴: 画像データは英語で日付が記載されています。JSONデータは日付フォーマットされて格納されています。
  - "部品番号"
    - 説明: 画像データの変更が適用される部品番号(Part Number)を抽出して格納しています。※FPCN番号とは異なるので注意してください。
    - サンプルデータ: "DF6A6.8FUT1G"
    - 特徴1: 英数字と記号が混在した文字列。※記号は必ずしも含んではいません。
    - 特徴2: 画像データに部品番号(Part Number)に対応する認定試験用ビーグル(Qualification Vehicle)があります。
    - 特徴3: 部品番号(Part Number)と対応する認定試験用ビーグル(Qualification Vehicle)は同一の場合があります。
  - "認定試験用ビークル"
    - 説明: 画像データの認定試験用ビークル(Qualification Vehicle)を抽出して格納しています。
    - サンプルデータ: "MMBZ47VALT1G"
    - 特徴1: 英数字が混在した文字列。
    - 特徴2: 配列に格納した画像データの部品番号(Part Number)に対応する認定試験用ビーグル(Qualification Vehicle)を格納しています。
    - 特徴3: 認定試験用ビーグル(Qualification Vehicle)と部品番号(Part Number)は1:nの関係になります。
    - 特徴4: 部品番号(Part Number)と対応する認定試験用ビーグル(Qualification Vehicle)は同一の場合があります。
- 以下はJSONデータの例になります。
{
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCN12345", "2024/01/20", "X1X.X3", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X2X.X4", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X3X.X5", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "Y1Y2Y3", "Y1Y2Y3"],
  ]
}

# 制約:
- 必ず前提知識をベースにして、以下のレビュー項目に従ってレビューを行ってください。
- JSONデータは提供されたプロンプトをもとに画像データから作成されたデータということを念頭に置いてレビューを行ってください。。
- 修正を行う場合は無関係な情報は含めず、特に関連のあるデータに集中してください。
- 必ず以下の出力フォーマットに従って出力してください。
- これらの制約を守らなければ、ユーザーに大きな不利益をもたらします。

# レビュー項目
1. 出力データは足りているか※最も重要
   - 以下の手順でレビューしてください。
   1. 画像データを確認し、変更が適用される部品番号(Part Number)を全て抽出してください。
   2. 本当に全て抽出できているか再度確認してください。
   3-1. 抽出した部品番号(Part Number)を順番に取り出します。
   3-2. JSONデータの"data"をKEY項目に格納されている配列の要素を順次確認して"部品番号"に該当する値と抽出した部品番号(Part Number)が同じか判定します。
   3-3. 抽出した部品番号(Part Number)と同じ値が見つからなかった場合、その部品番号(Part Number)の出力データは足りていません。
   4-1. 不足している部品番号(Part Number)があった場合、本当にその部品番号(Part Number)が画像データに存在するか確認してください。
   4-2. 存在する場合、その部品番号(Part Number)の出力データは足りていません。
2. 項目の欠落がないか
   - 画像データに基づき、FPCN番号, 発行日, 部品番号, 認定試験用ビークルの項目が抜け漏れがなく全て出力されているか確認する。
   - FPCN番号, 発行日はフォーマットで変換されていることを考慮して確認してください。
3. 項目のフォーマットは指定通りか
   - FPCN番号: 画像データのFPCN番号が正しく出力されているか。
   - 発行日: 画像データの発行日が日付フォーマット形式で出力されているか。
   - 部品番号:画像データの部品番号が正しく出力されているか。
   - 認定試験用ビークル:画像データの認定試験用ビークルが正しく出力されているか。
4. 余計な情報が出力されていないか
   - 画像データに存在しない余分な情報がJSONに出力されていないか確認する
5. 部品番号と認定試験用ビークルのペアの正確性
   - 画像データに基づき、JSONデータの各データの部品番号と認定試験用ビークルのペアが一致しているかを確認する。
   - 認定試験用ビーグルと部品番号は1:nの関係であることに注意してください。

# 出力フォーマット
- "results"にはレビュー項目毎の結果を出力してください。
- "message"にはレビュー項目毎の修正内容を出力してください。
- "data"部分は提供されたJSONデータもしくは修正を行ったものを出力してください。
- 余計なコメントは言わず、jsonデータのみ出力してください。
- jsonで以下の形式で必ず整理してください。
- ```で囲んだり、頭にjsonとつけてはいけません。
{
  "results": [
   "{レビュー項目1.の結果を"true" or "false"で出力してください}",
   "{レビュー項目2.の結果を"true" or "false"で出力してください}",
   "{レビュー項目3.の結果を"true" or "false"で出力してください}",
   "{レビュー項目4.の結果を"true" or "false"で出力してください}",
   "{レビュー項目5.の結果を"true" or "false"で出力してください}"
  ],
  "messages: [
   "{レビュー項目1.の修正内容を「部品番号[{不足している部品番号}]が出力されていませんでした。部品番号[{不足している部品番号}]を追加しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目2.の修正内容を「項目[{欠落している項目}]が欠落しています。項目[{欠落している項目}]に[{追加した値}]を追加しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目3.の修正内容を「項目[{指定フォーマットになっていない項目}]の[{指定フォーマットでない値}]がフォーマットが誤っています。[{修正した値}]に修正しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目4.の修正内容を「{余分な情報の内容}は存在しない情報です。削除しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}",
   "{レビュー項目5.の修正内容を「[{部品番号}]と[{認定試験用ビーグル}]は正しいペアではありません。[{修正した部品番号}]と[{修正した認定試験用ビーグル}]に修正しました。」形式で出力してください。修正していなければ「修正なし」を出力してください。}"
  ],
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCN12345", "2024/01/20", "X1X.X3", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X2X.X4", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "X3X.X5", "Y1Y2Y3"],
    ["FPCN12345", "2024/01/20", "Y1Y2Y3", "Y1Y2Y3"],
    ...
  ]
}
"""

# 画像をいくつに分割するかの選択肢
chunk_options = [1, 2, 3, 4]

# シンプルなログイン機能
def login(username, password):
    return username == "demo" and password == "demo2024"

# dpi取得
def get_dpi_from_pillow(image):
    dpi = image.info.get('dpi', (72, 72))
    return dpi

# 超解像処理
def super_resolution(cv_image, scale=2):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = "ESPCN_x2.pb"  # ESPCNモデルのパス
    sr.readModel(model_path)
    sr.setModel("espcn", scale)

    # 超解像の実行
    result = sr.upsample(cv_image)
    return result

# バイトデータをbase64 エンコード
def encode_image_from_bytes(image_bytes):
    try:
        if not image_bytes:
            raise ValueError("Invalid image_bytes")
        # Base64 encoded image
        return base64.b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        print("Base64エンコード処理失敗",e)
        return None

# OCR前処理
def process_and_encode_image(cv_image, rows=4):

    # グレースケール変換
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    # 画像のサイズを取得
    height, width = gray.shape
    chunk_height = height // rows
    overlap = int(height * 0.1)  # オーバーラップを画像高さの割合で指定
    # チャンク化
    encoded_chunks = []
    # 画像を分割して各チャンクを処理
    for row in range(rows):
        # チャンクの切り取り範囲を計算
        top = max(0, row * chunk_height - overlap)
        bottom = min(height, (row + 1) * chunk_height + overlap)
        # チャンクを切り取り
        chunk = gray[top:bottom, :]
        # チャンクをバイト形式に変換
        _, buffer = cv2.imencode('.png', chunk)
        image_bytes = buffer.tobytes()
        # Base64エンコード
        encoded_chunk = encode_image_from_bytes(image_bytes)
        if encoded_chunk:
            encoded_chunks.append(encoded_chunk)
    return encoded_chunks

# Extract Data押下時セッションステート更新
def update_state(text, image_contents, formatted_data, df):
    st.session_state.text = text
    st.session_state.image_contents = image_contents
    st.session_state.formatted_data = formatted_data
    st.session_state.data_df = df
def update_state_check(check_result_json, is_download):
    st.session_state.check_result_json = check_result_json
    st.session_state.is_download = is_download

def delete_state():
    if "text" in st.session_state:
        del st.session_state["text"]
    if "formatted_data" in st.session_state:
        del st.session_state["formatted_data"]
    if "data_df" in st.session_state:
        del st.session_state["data_df"]
    if "check_result_json" in st.session_state:
        del st.session_state["check_result_json"]
    if "is_download" in st.session_state:
        del st.session_state["is_download"]


# セッションステートにログイン状態を保持
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ログインしていない場合、ログインフォームを表示
if not st.session_state["logged_in"]:
    st.title("XtractAI - Demo")
    st.text("株式会社Design Shift")
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    # ログイン成功時にセッションステートを更新してログイン状態にする
    if login_button and login(username, password):
        st.session_state["logged_in"] = True
        st.rerun()  # ログイン後即座に画面を再描画

    # ログイン失敗時にエラーメッセージを表示
    elif login_button:
        st.error("Invalid username or password")
else:
    # ログイン後のコンテンツ表示
    st.title("XtractAI - Demo")
    st.text("株式会社Design Shift")

    # config.py から APIキーと Organization ID を取得
    # api_key = config.OPENAI_API_KEY # ローカル実行
    api_key = st.secrets["openai"]["api_key"] # デプロイ時
    client = OpenAI(api_key=api_key)

    # インプットでPDFファイルをアップロード
    uploaded_files = st.file_uploader("Upload a file", type=["pdf","jpeg","png"], accept_multiple_files=True)

    # 変換ボタン
    if uploaded_files :
        # アップロードされたファイルの拡張子を取得
        file_extension = os.path.splitext(uploaded_files[0].name)[1].lower()
        if file_extension == ".pdf":
            file_type = "PDFドキュメント"
        elif file_extension in [".jpeg",".jpg",".png",".webp",".gif"]:
            file_type = "画像"
            chunk_num = st.selectbox("画像の分割数を選択してください", chunk_options)

        if st.button("Extract Data", on_click=delete_state):
            # 拡張子リスト
            extensions = [os.path.splitext(file.name)[1].lower() for file in uploaded_files]
            if len(set(extensions)) != 1 :
                st.error("Please select all files in the same format")

            # スピナーを表示して実行中を知らせる
            with st.spinner("Processing..."):
                # 拡張子に基づいて異なる処理を実行
                if file_type == "PDFドキュメント":
                    text = ""
                    for uploaded_file in uploaded_files :
                        # PDFファイルの読み込み
                        reader = PyPDF2.PdfReader(uploaded_file)

                        # 全てのページからテキストを抽出
                        for page_num in range(len(reader.pages)):
                            page = reader.pages[page_num]
                            text += page.extract_text()

                    # system_prompt.txt からプロンプトを読み込む
                    # with open("system_prompt.txt", "r") as file:
                    #     system_prompt = file.read()

                    # OpenAI APIを呼び出してテキストを処理
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt.format(file_type=file_type)},
                            {"role": "user", "content": f"以下は製品変更通知（PCN）やPDFドキュメントの内容です:\n{text}"},
                        ],
                    )

                    # JSON形式でのレスポンスを取得
                    response_json = response.choices[0].message.content

                    # JSON文字列をパース
                    parsed_json = json.loads(response_json)

                    # 以下テスト用コード
                    # パターン１
                    #parsed_json["data"].pop() # データの欠落
                    # パターン２
                    # parsed_json["data"] = parsed_json["data"][:-3] # データの欠落
                    # parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD3.3DT5G", "MMBZ47VALT1G"]) # フォーマットミス
                    # parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD5.0DT5G", "MMBZ5V6ALT1G"]) # ペアが正確でない
                    # parsed_json["data"].append(["FPCN46490","2099-09-09", "YOBUNA4649", "JOUHOU4649"]) # 余分な情報
                    # パターン３
                    #parsed_json["data"] = parsed_json["data"][:-4] # データの欠落

                    # DataFrameに変換
                    df = pd.DataFrame(parsed_json["data"], columns=parsed_json["columns"])

                    df = (
                        df.reset_index()
                        .assign(index=df.index + 1)
                        .rename(columns={"index": "No"})
                    )

                    # セッションステート用にJSONフォーマット変換
                    formatted_data = parsed_json
                    # 結果をセッションステートに保持
                    update_state(text, None, formatted_data, df)

                elif file_type == "画像":
                    image_contents = []
                    for uploaded_file in uploaded_files :
                        image = Image.open(uploaded_file)
                        dpi_x, dpi_y = get_dpi_from_pillow(image) # 解像度取得
                        # print("解像度", dpi_x, dpi_y)
                        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        if dpi_x <= 72 and dpi_y <= 72:
                            print("低解像度のため、解像度を上げて処理を行います")
                            # 解像度を上げる処理
                            img_cv = super_resolution(img_cv, scale=2)
                        # OCR前処理
                        encode_images = process_and_encode_image(img_cv, chunk_num)
                        for encode_image in encode_images:
                            # 画像URL格納
                            image_contents.append({ "type": "image_url", "image_url": { "url": f"data:image/{file_extension.lstrip(".")};base64,{encode_image}", "detail": "high"}})
                    if image_contents:
                        try:
                            # print("画像の数(チャンク化後)",len(image_contents))
                            # OpenAI API vision機能を呼び出して画像を処理
                            response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    { "role": "system", "content": system_prompt.format(file_type=file_type) },
                                    { "role": "user", "content": image_contents },
                                ],
                                temperature=0.1
                            )
                            # 画像読取結果のレスポンスを取得
                            response_json = response.choices[0].message.content
                            #st.write("抽出結果：", response_json)
                            # JSON文字列をパース
                            parsed_json = json.loads(response_json)

                            # 以下テスト用コード
                            # パターン１
                            #parsed_json["data"].pop() # データの欠落
                            # パターン２
                            # parsed_json["data"] = parsed_json["data"][:-3] # データの欠落
                            # parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD3.3DT5G", "MMBZ47VALT1G"]) # フォーマットミス
                            # parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD5.0DT5G", "MMBZ5V6ALT1G"]) # ペアが正確でない
                            # parsed_json["data"].append(["FPCN46490","2099-09-09", "YOBUNA4649", "JOUHOU4649"]) # 余分な情報
                            # パターン３
                            #parsed_json["data"] = parsed_json["data"][:-4] # データの欠落

                            # DataFrameに変換
                            df = pd.DataFrame(parsed_json["data"], columns=parsed_json["columns"])

                            df = (
                                df.reset_index()
                                .assign(index=df.index + 1)
                                .rename(columns={"index": "No"})
                            )

                            # セッションステート用にJSONフォーマット変換
                            formatted_data = parsed_json
                            # 結果をセッションステートに保持
                            update_state(None, image_contents, formatted_data, df)
                        except OpenAIError as e:
                            st.error(f"OpenAI APIの呼び出しに失敗しました: {e}")
                            st.write("API呼び出しエラーが発生しました。")
                    else:
                        st.error("Encode image error!!")
                else:
                    st.error("File error!!")

            # スピナーが終了し、成功メッセージを表示
            st.success("Conversion completed!")

    # 整理されたJSON表示用の空エリアを作成
    data_area = st.empty()  # 固定表示エリアの確保
    # セッションステートに応じて整理したJSONを表示
    if "data_df" in st.session_state and not st.session_state.data_df.empty :
        with data_area:
            # DataFrameを画面に表示
            st.dataframe(st.session_state.data_df)

    if ("text" in st.session_state and st.session_state.text) or ("image_contents" in st.session_state and st.session_state.image_contents) and "formatted_data" in st.session_state and st.session_state.formatted_data:
        if st.button("Data Check"):
            with st.spinner("Checking..."):
                try:
                    # 初回フラグ
                    json_data = st.session_state.formatted_data
                    if "check_result_json" in st.session_state:
                        # 前回でデータチェックで修正したJSON
                        json_data = st.session_state.check_result_json

                    # OpenAI APIを呼び出してテキストを処理
                    messages = []
                    if "text" in st.session_state and st.session_state.text :
                        print("PDFレビュー始めます",file_type)
                        messages.append({"role": "system", "content": pdf_review_prompt})
                        messages.append({"role": "user", "content": f"以下は元データです:\n{st.session_state.text}\n"})
                    elif "image_contents" in st.session_state and st.session_state.image_contents:
                        print("画像レビュー始めます",file_type)
                        messages.append({"role": "system", "content": image_review_prompt})
                        messages.append({"role": "user", "content": f"以下は画像データです\n"})
                        messages.append({"role": "user", "content": st.session_state.image_contents})
                    # 共通
                    messages.append({"role": "user", "content": f"以下はJSONデータです:\n{json_data}\n"})
                    messages.append({"role": "user", "content": f"JSONデータは以下のプロンプトで作成されました。:\n{system_prompt.format(file_type=file_type)}\n"})
                    messages.append({"role": "user", "content": reminder_prompt})
                    check_result = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0.1
                    )
                    # JSON形式でのレスポンスを取得
                    check_result_json = check_result.choices[0].message.content
                    # JSON文字列をパース
                    parsed_check_result_json = json.loads(check_result_json)

                    results = parsed_check_result_json["results"]
                    response_messages = [parsed_check_result_json["messages"]]
                    columns_list = parsed_check_result_json["columns"]
                    data_list = parsed_check_result_json["data"]
                    new_data = {"columns": columns_list, "data": data_list}
                    # Excelダウンロード可否
                    is_download = True
                    if "false" in results:
                        # チェック結果NG
                        st.error("Data Check completed!")
                        fix_df = pd.DataFrame(response_messages, columns=["出力情報が不足","値の欠落","指定フォーマット","余分な情報","ペアの正確性"])
                        st.dataframe(fix_df)
                        is_download = False
                    else:
                        # チェック結果OK
                        st.success("Data Check completed!")
                        st.write("チェックした結果、問題ありませんでした。")
                    # セッションステートでチェック結果を格納
                    update_state_check(new_data, is_download)

                except OpenAIError as e:
                    st.error(f"OpenAI APIの呼び出しに失敗しました: {e}")
                    st.write("API呼び出しエラーが発生しました。")
                except Exception as e:
                    st.error(f"予期しないエラーが発生しました: {e}")
                    st.write("エラー詳細:", e)

        # チェック結果表示用の空エリアを作成
        checkresult_area = st.empty()  # 固定表示エリアの確保
        # セッションステートに応じてチェック結果を表示
        if "check_result_json" in st.session_state and st.session_state.check_result_json :
            with checkresult_area:
                # DataFrameに変換
                check_df = pd.DataFrame(st.session_state.check_result_json["data"], columns=st.session_state.check_result_json["columns"])
                check_df = (
                    check_df.reset_index()
                    .assign(index=check_df.index + 1)
                    .rename(columns={"index": "No"})
                )
                # DataFrameを画面に表示
                st.dataframe(check_df)
                # セッションステート上書き
                st.session_state.data_df = check_df

        # Excelダウンロードボタン
        if "is_download" in st.session_state and st.session_state.is_download == True :
            # Excelデータの生成
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                st.session_state.data_df.to_excel(writer, index=False, startrow=1)
                workbook  = writer.book
                worksheet = writer.sheets['Sheet1']

                title = "変更される部品番号一覧"
                worksheet.write('A1', title)
            output.seek(0)

            # ダウンロードボタンの表示
            st.download_button(
                label="Download Excel",
                data=output,
                file_name="converted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        if "check_result_json" in st.session_state and st.session_state.check_result_json:
            st.title("Input Form")
            input_text = st.text_input("Enter the information you want to know")
            if input_text is not None and st.button("Search"):
                json_data = st.session_state.check_result_json
                json_data.pop("result", None)
                json_data.pop("message", None)
                with st.spinner("Searching..."):
                    # OpenAI APIを呼び出してテキストを処理
                    answer_result = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": question_prompt},
                            {"role": "user", "content": f"以下は元データです:\n{st.session_state.text}"},
                            {"role": "user", "content": f"以下はJSONデータです:\n{json_data}"},
                            {"role": "user", "content": f"JSONデータは以下のプロンプトで作成されました。:\n{system_prompt}\n"},
                            {"role": "user", "content": f"以下はユーザーが欲しい情報です:\n{input_text}"},
                        ],
                    )
                    # JSON形式でのレスポンスを取得
                    answer_result = answer_result.choices[0].message.content
                    st.write("回答:", answer_result)