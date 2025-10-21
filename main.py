import asyncio
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api import logger
from astrbot.core.star.star_tools import StarTools

@register("astrbot_plugin_ragflow_adapter", "RC-CHN", "使用RAGFlow检索增强生成", "v0.1")
class RAGFlowAdapterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.plugin_data_dir: Path = StarTools.get_data_dir()

        # 初始化配置变量
        self.ragflow_base_url = ""
        self.ragflow_api_key = ""
        self.ragflow_kb_ids = []
        self.enable_query_rewrite = False
        self.query_rewrite_provider_key = ""

    def _mask_sensitive_info(self, info: str, keep_last: int = 6) -> str:
        """隐藏敏感信息，只显示最后几位。"""
        if not isinstance(info, str) or len(info) <= keep_last:
            return info
        return f"******{info[-keep_last:]}"

    async def initialize(self):
        """
        初始化插件，加载并打印配置。
        """
        self.plugin_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        self.ragflow_base_url = self.config.get("ragflow_base_url", "")
        self.ragflow_api_key = self.config.get("ragflow_api_key", "")
        self.ragflow_kb_ids = self.config.get("ragflow_kb_ids", [])
        self.enable_query_rewrite = self.config.get("enable_query_rewrite", False)
        self.query_rewrite_provider_key = self.config.get("query_rewrite_provider_key", "")

        # 打印日志
        logger.info("RAGFlow 适配器插件已初始化。")
        logger.info(f"  RAGFlow API 地址: {self.ragflow_base_url}")
        logger.info(f"  RAGFlow API Key: {self._mask_sensitive_info(self.ragflow_api_key)}")
        
        masked_kb_ids = [self._mask_sensitive_info(str(kid)) for kid in self.ragflow_kb_ids]
        logger.info(f"  RAGFlow 知识库 ID: {masked_kb_ids}")
        
        logger.info(f"  启用查询重写: {'是' if self.enable_query_rewrite else '否'}")
        if self.enable_query_rewrite:
            logger.info(f"  查询重写 Provider: {self.query_rewrite_provider_key or '未指定'}")


    async def _rewrite_query(self, original_query: str) -> str:
        """
        使用指定的大语言模型优化用户查询，以获得更好的检索结果。
        如果未启用或未配置，则返回原始查询。
        """
        if not self.enable_query_rewrite:
            return original_query

        if not self.query_rewrite_provider_key:
            logger.warning("查询重写已启用，但未选择 Provider。跳过重写。")
            return original_query

        provider = self.context.get_provider_by_id(self.query_rewrite_provider_key)
        if not provider:
            logger.error(f"找不到用于查询重写的 Provider (ID: '{self.query_rewrite_provider_key}')。跳过重写。")
            return original_query

        prompt = f"请优化以下用户问题，使其更适合作为检索系统的输入。请直接返回优化后的问题，不要包含任何解释或引言。\n原始问题：{original_query}\n优化后的问题："
        
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

    async def _query_ragflow(self, query: str) -> str:
        """
        (占位符) 使用给定的查询与 RAGFlow API 进行交互。
        此方法需要被实现。
        """
        if not all([self.ragflow_base_url, self.ragflow_api_key, self.ragflow_kb_ids]):
            return "RAGFlow 未完全配置，请检查插件设置。"

        # TODO: 在此处实现对 RAGFlow 的真实 API 调用
        # 通常你会使用 'httpx' 或 'aiohttp' 这样的库
        logger.info(f"正在查询 RAGFlow: URL={self.ragflow_base_url}, KB_IDs={self.ragflow_kb_ids}, Query='{query}'")
        
        # 这是一个占位符响应
        await asyncio.sleep(1) # 模拟网络延迟
        return f"这是一个来自 RAGFlow 的占位符响应，查询内容为: '{query}'"

    @filter.command("retrieve")
    async def retrieve_command(self, event: AstrMessageEvent, content: str):
        """
        使用 RAGFlow 执行 RAG 查询。
        用法: /retrieve <你的问题>
        """
        if not content:
            yield event.plain_result("请输入您的问题。用法: /retrieve <你的问题>")
            return

        yield event.plain_result("正在处理您的问题...")

        # 1. 如果启用，则重写查询
        final_query = await self._rewrite_query(content)
        if final_query != content:
            yield event.plain_result(f"优化后的问题: {final_query}")

        # 2. 查询 RAGFlow
        response = await self._query_ragflow(final_query)

        # 3. 发送响应
        yield event.plain_result(response)

    async def terminate(self):
        """
        插件卸载时清理资源。
        """
        logger.info("RAGFlow 适配器插件已终止。")