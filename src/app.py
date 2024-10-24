import streamlit as st
from openai import OpenAI
import os
import config
import PyPDF2
import json
import pandas as pd
import io  # メモリ上にファイルを保存するために必要

system_prompt = """
# 役割:
- あなたは提供された製品変更通知（PCN）やPDFドキュメントの内容を、以下のキーに従って構造化し、整理されたjsonデータを出力します。

# 制約:
- 各キーには該当する情報を正確に埋めてください。
- PDFにその情報が含まれていない場合は「N/A」と記載します。
- 提供された情報を以下のキーの順に従って整理してください。
- 無関係な情報は含めず、特に関連のあるデータに集中してください。

# キー構造:
1. FPCN番号:  
   - 説明: 各文書に付与された識別番号を記載。
   - フォーマット: FPCNXXXXX（例: FPCN12345）

2. 発行日:  
   - 説明: 文書が発行された日付を記載。
   - フォーマット: YYYY/MM/DD（例: 2023/12/01）

3. 部品番号(Part Number):  
   - 説明: 変更が適用される部品番号を記載。

4. 認定試験用ビークル(Qualification Vehicle)
   - 説明: 品質や性能を確認するためのテスト用部品の番号を記載

# 出力フォーマット
余計なコメントは言わず、jsonデータのみ出力してください。
jsonで以下の形式で必ず整理してください。
```で囲んだり、頭にjsonとつけてはいけません。
{
  "columns": ["FPCN番号", "発行日", "部品番号", "認定試験用ビークル"],
  "data": [
    ["FPCNXXXXX", "2024/01/20", "XXX", "YYY"],
    ...
  ]
}
"""


# シンプルなログイン機能
def login(username, password):
    return username == "demo" and password == "demo2024"


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
    # api_key = config.OPENAI_API_KEY
    api_key = st.secrets["openai"]["api_key"]
    client = OpenAI(api_key=api_key)

    # インプットでPDFファイルをアップロード
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    # 変換ボタン
    if uploaded_file is not None:
        if st.button("Extract Data"):
            # スピナーを表示して実行中を知らせる
            with st.spinner("Processing..."):
                # PDFファイルの読み込み
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""

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
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                )

                # JSON形式でのレスポンスを取得
                response_json = response.choices[0].message.content

                # JSON文字列をパース
                parsed_json = json.loads(response_json)

                # DataFrameに変換
                df = pd.DataFrame(parsed_json["data"], columns=parsed_json["columns"])

                df = (
                    df.reset_index()
                    .assign(index=df.index + 1)
                    .rename(columns={"index": "No"})
                )

                # DataFrameを画面に表示
                st.dataframe(df)

                # Excelデータの生成
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False)
                output.seek(0)

            # スピナーが終了し、成功メッセージを表示
            st.success("Conversion completed!")

            # ダウンロードボタンの表示
            st.download_button(
                label="Download Excel",
                data=output,
                file_name="converted_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
