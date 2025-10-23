import asyncio
import httpx
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import RAGFlowAdapterPlugin
    from astrbot.api.provider import ProviderRequest


def mask_sensitive_info(info: str, keep_last: int = 6) -> str:
    """隐藏敏感信息，只显示最后几位。"""
    if not isinstance(info, str) or len(info) <= keep_last:
        return info
    return f"******{info[-keep_last:]}"


def inject_content_into_request(plugin: "RAGFlowAdapterPlugin", req: "ProviderRequest", content: str):
    """
    根据配置，将指定内容注入到 ProviderRequest 对象中。
    """
    if not content:
        return

    # 统一的 RAG 内容模板
    rag_prompt_template = f"--- 以下是参考资料 ---\n{content}\n--- 请根据以上资料回答问题 ---"

    if plugin.rag_injection_method == "user_prompt":
        req.prompt = f"{rag_prompt_template}\n\n{req.prompt}"
        logger.debug("RAG content injected into user_prompt.")
    elif plugin.rag_injection_method == "insert_system_prompt":
        # 插入到倒数第二的位置，确保在用户最新消息之前
        req.contexts.insert(-1, {"role": "system",
                                 "content": rag_prompt_template})
        logger.debug("RAG content inserted as a new system message.")
    else:  # 默认为 system_prompt
        if req.system_prompt:
            req.system_prompt = f"{req.system_prompt}\n\n{rag_prompt_template}"
        else:
            req.system_prompt = rag_prompt_template
        logger.debug("RAG content injected into system_prompt.")


async def query_ragflow(plugin: "RAGFlowAdapterPlugin", query: str) -> str:
    """
    使用给定的查询与 RAGFlow API 进行交互，并返回拼接好的上下文。
    """
    if not all([plugin.ragflow_base_url, plugin.ragflow_api_key, plugin.ragflow_kb_ids]):
        logger.warning("RAGFlow 未完全配置，跳过检索。")
        return ""

    url = f"{plugin.ragflow_base_url.rstrip('/')}/api/v1/retrieval"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {plugin.ragflow_api_key}",
    }
    data = {
        "question": query,
        "dataset_ids": plugin.ragflow_kb_ids,
        "top_k": 5,
        "similarity_threshold": 0.35
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()

            api_data = response.json()
            if api_data.get("code") != 0:
                logger.error(f"RAGFlow API 返回错误: {api_data}")
                return ""

            chunks = api_data.get("data", {}).get("chunks", [])
            if not chunks:
                logger.info("RAGFlow 未检索到相关内容。")
                return ""

            retrieved_content = "\n\n".join(
                [chunk.get("content", "") for chunk in chunks])
            logger.info(f"成功从 RAGFlow 检索到 {len(chunks)} 条内容。")
            logger.debug(f"检索到的内容: \n{retrieved_content}")
            return retrieved_content

    except httpx.RequestError as e:
        logger.error(f"请求 RAGFlow API 时出错: {e}", exc_info=True)
        return ""
    except Exception as e:
        logger.error(f"处理 RAGFlow 响应时发生未知错误: {e}", exc_info=True)
        return ""


async def archive_conversation(plugin: "RAGFlowAdapterPlugin", event: AstrMessageEvent):
    """
    将当前会话的近期对话历史进行归档。
    """
    logger.info(f"触发了会话 {event.get_session_id()} 的归档流程。")
    # TODO: 实现获取对话历史、总结、上传的逻辑
    await asyncio.sleep(1)  # 模拟异步操作
