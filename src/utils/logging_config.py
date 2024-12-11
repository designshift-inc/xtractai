import logging
import sys

def setup_logging():
    # ロガー作成
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 必要に応じてDEBUGやERRORに変更
    # フォーマッターを設定
    formatter1 = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter2 = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 標準出力用のハンドラー
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)  # ERROR未満のみ処理
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter1)
        logger.addHandler(stdout_handler)
    # エラー出力用のハンドラー
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stderr for h in logger.handlers):
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(formatter2)
        logger.addHandler(stderr_handler)
    return logger
