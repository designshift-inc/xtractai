import traceback
import streamlit as st
from openai import AzureOpenAI
# import config # ローカル実行
import PyPDF2
import json
import pandas as pd
import os
import io
import prompt.chat
import prompt.review
import prompt.xtract
from utils.logging_config import setup_logging  # メモリ上にファイルを保存するために必要
import prompt

logger = setup_logging()

# Extract Data押下時セッションステート更新
def update_state(text, formatted_data, df, publisher):
    st.session_state.text = text
    st.session_state.formatted_data = formatted_data
    st.session_state.data_df = df
    st.session_state.publisher = publisher
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
    if "publisher" in st.session_state:
        del st.session_state["publisher"]

def xtract_page():
    # ログイン後のコンテンツ表示
    st.title("XtractAI - Demo")
    st.text("株式会社Design Shift")

    # config.py から APIキーと Organization ID を取得
    #api_key = config.OPENAI_API_KEY # ローカル実行
    #api_key = st.secrets["openai"]["api_key"] # デプロイ時
    # azure_endpoint = config.AZURE_OPENAI_ENDPOINT # ローカル時
    # api_key = config.AZURE_OPENAI_KEY # ローカル時
    azure_endpoint = os.environ["OPENAI_URL"] # デプロイ時(環境変数)
    api_key = os.environ["OPENAI_KEY"] # デプロイ時(環境変数)
    # client = OpenAI(api_key=api_key)

    try:
        azure_client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-08-01-preview"
        )
    except Exception as e:
        # 他の予期しないエラー
        error_traceback = traceback.format_exc()
        logger.error(f"OpenAI Client Error:{e}")
        logger.error(error_traceback)
        st.error("OpenAI Client Error")

    # インプットでPDFファイルをアップロード
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    publisher_options = ["ON Semiconductor", "Texas Instruments"]
    # 発行元
    publisher = st.selectbox("発行元を選択してください", publisher_options)

    # 変換ボタン
    if uploaded_file is not None:
        if st.button("Extract Data", on_click=delete_state):
            # スピナーを表示して実行中を知らせる
            with st.spinner("Processing..."):
                # PDFファイルの読み込み
                reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                # PDFファイル名取得
                file_name = uploaded_file.name
                # 全てのページからテキストを抽出
                for page_num in range(len(reader.pages)):
                    text += f"--------------------------- PDF{page_num+1}ページ目開始 ---------------------------" + '\n'
                    page = reader.pages[page_num]
                    text += page.extract_text() + '\n'

                # system_prompt.txt からプロンプトを読み込む
                # with open("system_prompt.txt", "r") as file:
                #     system_prompt = file.read()

                if publisher == "ON Semiconductor":
                    logger.info("ON Semiconductorデータ抽出")
                    messages = [{"role": "system", "content": prompt.xtract.ON_SEMICONDUCTOR}]
                elif publisher == "Texas Instruments":
                    logger.info("Texas Instrumentsデータ抽出")
                    messages = [{"role": "system", "content": prompt.xtract.TEXAS_INSTRUMENTS}]
                #print(text)
                messages.append({"role": "user", "content": f"以下は製品変更通知（PCN）やPDFドキュメントの内容です:\n{text}"})
                messages.append({"role": "user", "content": prompt.xtract.REMINDER})
                # OpenAI APIを呼び出してテキストを処理
                try:
                    response = azure_client.chat.completions.create(
                        #model="gpt-4o-mini",
                        model="gpt-4o",
                        messages=messages,
                        temperature=0
                    )
                    # JSON形式でのレスポンスを取得
                    response_json = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                    # JSON文字列をパース
                    parsed_json = json.loads(response_json)

                    # 以下デモ用コード
                    if "Final_FPCN20579XA.pdf" == file_name :
                        parsed_json["data"] = parsed_json["data"][:-3] # データの欠落
                        parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD3.3DT5G", "MMBZ47VALT1G"]) # フォーマットミス
                        parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD5.0DT5G", "MMBZ5V6ALT1G"]) # ペアが正確でない
                        parsed_json["data"].append(["FPCN46490","2099-09-09", "YOBUNA4649", "JOUHOU4649"]) # 余分な情報

                    # 以下テスト用コード
                    # パターン１
                    #parsed_json["data"].pop() # データの欠落
                    # パターン２
                    # parsed_json["data"] = parsed_json["data"][:-3] # データの欠落
                    # parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD3.3DT5G", "MMBZ47VALT1G"]) # フォーマットミス
                    # parsed_json["data"].append(["FPCN20579","2016-06-28", "UESD5.0DT5G", "MMBZ5V6ALT1G"]) # ペアが正確でない
                    # parsed_json["data"].append(["FPCN46490","2099-09-09", "YOBUNA4649", "JOUHOU4649"]) # 余分な情報
                    # parsed_json["data"] = parsed_json["data"][:-2] # データの欠落
                    # parsed_json["data"].append(["PCN20240729003.0","2023-06-01", "null", "LMR62014XMFE/NOPB"]) # フォーマットミス
                    # parsed_json["data"].append(["PCN20240729003.0","2024-07-01", "null", "TIGAUPEA1234"]) # ペアが正確でない
                    # parsed_json["data"].append(["FPCN46490","2099-09-09", "YOBUNA4649", "JOUHOU4649"]) # 余分な情報
                    # パターン３
                    #parsed_json["data"] = parsed_json["data"][:-9] # データの欠落

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
                    update_state(text, formatted_data, df, publisher)
                    # スピナーが終了し、成功メッセージを表示
                    st.success("Conversion completed!")
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Extract Data Decode Error:{e}")
                    logger.error(response_json)
                    if finish_reason == "length":
                        st.error("Data extraction limit exceeded")
                    else:
                        st.error("Decode error")
                except Exception as e:
                    # 他の予期しないエラー
                    error_traceback = traceback.format_exc()
                    logger.error(f"Extract Data Error:{e}")
                    logger.error(error_traceback)
                    st.error("Conversion error")

    # 整理されたJSON表示用の空エリアを作成
    data_area = st.empty()  # 固定表示エリアの確保
    # セッションステートに応じて整理したJSONを表示
    if "data_df" in st.session_state and not st.session_state.data_df.empty :
        with data_area:
            # DataFrameを画面に表示
            st.dataframe(st.session_state.data_df)

    if "text" in st.session_state and st.session_state.text and "formatted_data" in st.session_state and st.session_state.formatted_data:
        if st.button("Data Check"):
            with st.spinner("Checking..."):
                try:
                    # 初回フラグ
                    json_data = st.session_state.formatted_data
                    if "check_result_json" in st.session_state:
                        # 前回でデータチェックで修正したJSON
                        json_data = st.session_state.check_result_json

                    # OpenAI APIを呼び出してテキストを処理
                    if st.session_state.publisher == "ON Semiconductor":
                        logger.info("ON Semiconductorレビュー")
                        xtract_prompt = prompt.xtract.ON_SEMICONDUCTOR
                        review_prompt = prompt.review.ON_SEMICONDUCTOR
                    elif st.session_state.publisher == "Texas Instruments":
                        logger.info("Texas Instrumentsレビュー")
                        xtract_prompt = prompt.xtract.TEXAS_INSTRUMENTS
                        review_prompt = prompt.review.TEXAS_INSTRUMENTS
                    check_result = azure_client.chat.completions.create(
                        model="gpt-4o",
                        #model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": review_prompt},
                            {"role": "user", "content": f"以下はJSONデータです:\n{json_data}\n"},
                            {"role": "user", "content": f"以下は元データです:\n{st.session_state.text}\n"},
                            {"role": "user", "content": f"以下はプロンプトです:\n{xtract_prompt}\n"},
                            {"role": "user", "content": prompt.review.REMINDER},
                        ],
                        temperature=0.1
                    )
                    # JSON形式でのレスポンスを取得
                    check_result_json = check_result.choices[0].message.content
                    finish_reason = check_result.choices[0].finish_reason
                    # JSON文字列をパース
                    parsed_check_result_json = json.loads(check_result_json)
                    results = parsed_check_result_json["results"]
                    messages = [parsed_check_result_json["messages"]]
                    columns_list = parsed_check_result_json["columns"]
                    data_list = parsed_check_result_json["data"]
                    new_data = {"columns": columns_list, "data": data_list}
                    # Excelダウンロード可否
                    is_download = True
                    if "false" in results:
                        # チェック結果NG
                        st.error("Data Check completed!")
                        fix_df = pd.DataFrame(messages, columns=["出力情報が不足","値の欠落","指定フォーマット","余分な情報","ペアの正確性"])
                        st.dataframe(fix_df)
                        is_download = False
                    else:
                        # チェック結果OK
                        st.success("Data Check completed!")
                        st.write("チェックした結果、問題ありませんでした。")
                    # セッションステートでチェック結果を格納
                    update_state_check(new_data, is_download)
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"Data Check Decode Error:{e}")
                    logger.error(check_result_json)
                    if finish_reason == "length":
                        st.error("Data extraction limit exceeded")
                    else:
                        st.error("Decode error")
                except Exception as e:
                    # 他の予期しないエラー
                    error_traceback = traceback.format_exc()
                    logger.error(f"Data Check Error:{e}")
                    logger.error(error_traceback)
                    st.error("Data Check Error")

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
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    try:
                        st.session_state.data_df.to_excel(writer, index=False)
                    except AttributeError:
                        logger.error("DataFrame is missing or invalid")
                        raise
                output.seek(0)

                # ダウンロードボタンの表示
                st.download_button(
                    label="Download Excel",
                    data=output,
                    file_name="converted_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                # 他の予期しないエラー
                logger.error("Download Error")
                logger.error(f"{e}")
                st.error("Download Error")

        if "check_result_json" in st.session_state and st.session_state.check_result_json:
            st.title("Input Form")
            input_text = st.text_input("Enter the information you want to know")
            if input_text is not None and st.button("Search"):
                json_data = st.session_state.check_result_json
                json_data.pop("results", None)
                json_data.pop("messages", None)
                with st.spinner("Searching..."):
                    try:
                        # OpenAI APIを呼び出してテキストを処理
                        if st.session_state.publisher == "ON Semiconductor":
                            xtract_prompt = prompt.xtract.ON_SEMICONDUCTOR
                        elif st.session_state.publisher == "Texas Instruments":
                            xtract_prompt = prompt.xtract.TEXAS_INSTRUMENTS
                        answer_result = azure_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": prompt.chat.CHAT},
                                {"role": "user", "content": f"以下は元データです:\n{st.session_state.text}"},
                                {"role": "user", "content": f"以下はJSONデータです:\n{json_data}"},
                                {"role": "user", "content": f"JSONデータは以下のプロンプトで作成されました。:\n{xtract_prompt}\n"},
                                {"role": "user", "content": f"以下はユーザーが欲しい情報です:\n{input_text}"},
                            ],
                        )
                        # JSON形式でのレスポンスを取得
                        answer = answer_result.choices[0].message.content
                        finish_reason = answer_result.choices[0].finish_reason
                        st.write("回答:", answer)
                    except json.decoder.JSONDecodeError as e:
                        logger.error(f"Data Search Decode Error:{e}")
                        logger.error(answer)
                        if finish_reason == "length":
                            st.error("Data extraction limit exceeded")
                        else:
                            st.error("Decode error")
                    except Exception as e:
                        # 他の予期しないエラー
                        error_traceback = traceback.format_exc()
                        logger.error(f"Data Search Error:{e}")
                        logger.error(error_traceback)
                        st.error("Data Search Error")