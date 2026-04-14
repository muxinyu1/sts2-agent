import json
from typing import Any

import instructor
import inspect
from typing import Callable, Any, Optional

def generate_markdown_tools(functions: list[Callable]) -> str:
    """将 Python 函数列表转换为适合放入 Prompt 的 Markdown 文本。"""

    def annotation_to_str(annotation: Any) -> str:
        if annotation == inspect.Signature.empty or annotation == inspect.Parameter.empty:
            return "Any"
        if hasattr(annotation, "__name__") and str(annotation).startswith("<class '"):
            return annotation.__name__
        return str(annotation).replace("typing.", "")

    markdown_lines = []

    for i, func in enumerate(functions, 1):
        name = func.__name__
        doc = inspect.getdoc(func) or "No description."
        sig = inspect.signature(func)

        # 拆分 Docstring（主描述 / Args / Returns）
        main_desc = doc
        arg_descs = {}
        returns_desc = ""
        if "Args:" in doc:
            parts = doc.split("Args:")
            main_desc = parts[0].strip()
            args_and_more = parts[1]
            args_text = args_and_more.split("Returns:")[0]
            if "Returns:" in args_and_more:
                returns_desc = args_and_more.split("Returns:", 1)[1].strip().split("\n")[0]
            
            # 简单解析每行参数注释
            for line in args_text.split('\n'):
                line = line.strip()
                if ':' in line and not line.startswith('-'):
                    arg_name, arg_desc = line.split(':', 1)
                    arg_descs[arg_name.strip()] = arg_desc.strip()
        elif "Returns:" in doc:
            parts = doc.split("Returns:", 1)
            main_desc = parts[0].strip()
            returns_desc = parts[1].strip().split("\n")[0]

        # 写入工具名称和主要描述
        markdown_lines.append(f"### {i}. {name}")
        markdown_lines.append(f"- description: {main_desc}")

        # 解析并写入参数列表
        if not sig.parameters:
            markdown_lines.append("- params: none")
            markdown_lines.append("")
            continue

        markdown_lines.append("- params:")
        for param_name, param in sig.parameters.items():
            # 获取类型名称，处理 Optional/Union 等复杂类型
            annotation = param.annotation
            type_str = annotation_to_str(annotation)
                
            # 判断是否必填
            is_required = param.default == inspect.Parameter.empty
            req_str = "required" if is_required else "optional"
            
            # 获取参数的自然语言描述
            desc = arg_descs.get(param_name, "No description.")
            markdown_lines.append(f"  - {param_name}: {type_str}, {req_str}. {desc}")

        markdown_lines.append("")

    return "\n".join(markdown_lines)


def use_potion(slot: int, target: str | None = None) -> dict[str, Any]:
    """Action tool: use_potion.

    Args:
        slot: Potion slot index.
        target: Enemy entity_id for target-required potions.
    """
    raise NotImplementedError("Tool declaration placeholder: not implemented yet")
from langchain_core.utils.function_calling import convert_to_openai_tool
t = convert_to_openai_tool(use_potion)

print(generate_markdown_tools([use_potion]))