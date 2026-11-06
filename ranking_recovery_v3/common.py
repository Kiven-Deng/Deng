# v3 fix: support direct script execution; retain v2 model/data behavior and schema.
# 修正说明：
# 1. 所有产物写入唯一的新目录，不覆盖旧实验。
# 2. 保存文件指纹和运行配置，训练、评估可以核对数据来源。
# 3. 路径相对项目目录解析，避免不同启动目录导致文件错用。
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def new_directory(base, label):
    path = resolve(base) / (label + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
