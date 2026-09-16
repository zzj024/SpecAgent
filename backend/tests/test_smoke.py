"""冒烟测试：验证包结构、依赖安装、pytest 配置三件事都活着。

这不是业务测试，是环境的"心电图"——它挂了说明环境坏了，而不是代码逻辑有错。
"""

import fastapi
import pydantic
import psycopg
import pymupdf


def test_backend_packages_importable():
    """backend 下六个包都能被顶层 import（pythonpath 配置生效的证明）。"""
    import agents
    import api
    import memory
    import rag
    import reliability
    import safety

    assert {m.__name__ for m in (agents, api, memory, rag, reliability, safety)} == {
        "agents", "api", "memory", "rag", "reliability", "safety",
    }


def test_core_dependencies_importable():
    """M1 核心依赖可导入，且 Pydantic 是 v2（v1/v2 API 不兼容，必须钉死）。"""
    assert int(pydantic.VERSION.split(".")[0]) == 2
    assert fastapi.__version__
    assert psycopg.__version__
    assert pymupdf.__version__
