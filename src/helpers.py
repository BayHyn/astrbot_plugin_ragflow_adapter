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


async def rewrite_query(plugin: "RAGFlowAdapterPlugin", original_query: str) -> str:
    """
    使用指定的大语言模型优化用户查询，以获得更好的检索结果。
    如果未启用或未配置，则返回原始查询。
    """
    if not plugin.enable_query_rewrite:
        logger.debug("查询重写未启用，跳过。")
        return original_query

    if not plugin.query_rewrite_provider_key:
        logger.warning("查询重写已启用，但未选择 Provider。跳过重写。")
        return original_query

    provider = plugin.context.get_provider_by_id(
        plugin.query_rewrite_provider_key)
    if not provider:
        logger.error(
            f"找不到用于查询重写的 Provider (ID: '{plugin.query_rewrite_provider_key}')。跳过重写。")
        return original_query

    prompt = f"""你是一个为检索系统优化查询的专家。你的任务是将用户的日常对话问题，转换成一个简洁、充满关键词、适合向量检索的查询语句。

一个好的查询应该：
- 提取核心实体和术语。
- 移除无关的口语化表达（例如“你好”、“请问一下”）。
- 将模糊的指代（例如“那个东西”）转换为具体的名称。

这里有几个例子：

# 示例 1
原始问题：我们上次会议提到的那个新功能，关于用户自定义首页的，现在进度如何了？
优化后的问题：用户自定义首页功能开发进度与计划

# 示例 2
原始问题：最近有什么关于全球变暖对北极熊影响的研究吗？
优化后的问题：全球气候变暖对北极熊栖息地、繁殖和捕食行为的影响研究

# 示例 3
原始问题：公司新出的那个报销政策具体是怎么规定的？特别是差旅费方面。
优化后的问题：公司最新差旅费用报销政策标准与流程

---
现在，请处理以下问题。请直接返回优化后的问题，不要包含任何解释或引言。
原始问题：{original_query}
优化后的问题："""

    try:
        llm_resp = await provider.text_chat(prompt=prompt)
        if llm_resp and llm_resp.completion_text:
            rewritten_query = llm_resp.completion_text.strip()
            logger.info(f"查询已重写：'{original_query}' -> '{rewritten_query}'")
            return rewritten_query
        else:
            logger.warning("查询重写 Provider 返回了空内容，将使用原始查询。")
            return original_query
    except Exception as e:
        logger.error(f"查询重写时出错: {e}", exc_info=True)
        return original_query


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