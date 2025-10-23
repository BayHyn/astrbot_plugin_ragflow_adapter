import asyncio
import httpx
from pathlib import Path

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.api import logger
from astrbot.core.star.star_tools import StarTools
from astrbot.api.provider import ProviderRequest

from .src import helpers


@register("astrbot_plugin_ragflow_adapter", "RC-CHN", "使用RAGFlow检索增强生成", "v0.2")
class RAGFlowAdapterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.plugin_data_dir: Path = StarTools.get_data_dir()
        self.session_message_counts = {}

        # 初始化配置变量
        self.ragflow_base_url = ""
        self.ragflow_api_key = ""
        self.ragflow_kb_ids = []
        self.enable_query_rewrite = False
        self.query_rewrite_provider_key = ""
        self.rag_injection_method = "system_prompt"

        # 归档功能配置
        self.rag_archive_enabled = False
        self.rag_archive_dataset_id = ""
        self.rag_archive_threshold = 40
        self.rag_archive_summarize_enabled = False
        self.rag_archive_summarize_persona_id = ""
        self.rag_archive_summarize_provider_id = ""

    async def initialize(self):
        """
        初始化插件，加载并打印配置。
        """
        self.plugin_data_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        self.ragflow_base_url = self.config.get("ragflow_base_url", "")
        self.ragflow_api_key = self.config.get("ragflow_api_key", "")
        self.ragflow_kb_ids = self.config.get("ragflow_kb_ids", [])
        self.enable_query_rewrite = self.config.get(
            "enable_query_rewrite", False)
        self.query_rewrite_provider_key = self.config.get(
            "query_rewrite_provider_key", "")
        self.rag_injection_method = self.config.get(
            "rag_injection_method", "system_prompt")

        # 加载归档配置
        self.rag_archive_enabled = self.config.get("rag_archive_enabled", False)
        self.rag_archive_dataset_id = self.config.get("rag_archive_dataset_id", "")
        self.rag_archive_threshold = self.config.get("rag_archive_threshold", 40)
        self.rag_archive_summarize_enabled = self.config.get("rag_archive_summarize_enabled", False)
        self.rag_archive_summarize_persona_id = self.config.get("rag_archive_summarize_persona_id", "")
        self.rag_archive_summarize_provider_id = self.config.get("rag_archive_summarize_provider_id", "")

        # 打印日志
        logger.info("RAGFlow 适配器插件已初始化。")
        logger.info(f"  RAGFlow API 地址: {self.ragflow_base_url}")
        logger.info(
            f"  RAGFlow API Key: {helpers.mask_sensitive_info(self.ragflow_api_key)}")

        masked_kb_ids = [helpers.mask_sensitive_info(
            str(kid)) for kid in self.ragflow_kb_ids]
        logger.info(f"  RAGFlow 知识库 ID: {masked_kb_ids}")

        logger.info(f"  启用查询重写: {'是' if self.enable_query_rewrite else '否'}")
        if self.enable_query_rewrite:
            logger.info(
                f"  查询重写 Provider: {self.query_rewrite_provider_key or '未指定'}")
        logger.info(f"  RAG 内容注入方式: {self.rag_injection_method}")

        # 打印归档配置日志
        logger.info(f"  启用自动归档: {'是' if self.rag_archive_enabled else '否'}")
        if self.rag_archive_enabled:
            logger.info(f"    归档数据集 ID: {self.rag_archive_dataset_id}")
            logger.info(f"    归档消息阈值: {self.rag_archive_threshold}")
            logger.info(f"    归档前总结: {'是' if self.rag_archive_summarize_enabled else '否'}")
            if self.rag_archive_summarize_enabled:
                logger.info(f"      总结 Persona: {self.rag_archive_summarize_persona_id or '未指定'}")
                logger.info(f"      总结 Provider: {self.rag_archive_summarize_provider_id or '未指定'}")

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """
        在 LLM 请求前，自动执行 RAG 检索并注入上下文。
        """
        # 1. 重写查询
        final_query = await helpers.rewrite_query(self, req.prompt)

        # 2. 查询 RAGFlow
        rag_content = await helpers.query_ragflow(self, final_query)

        # 3. 注入内容
        if rag_content:
            helpers.inject_content_into_request(self, req, rag_content)

        # 4. 处理自动归档逻辑
        if self.rag_archive_enabled:
            session_id = event.get_session_id()
            count = self.session_message_counts.get(session_id, 0) + 1
            self.session_message_counts[session_id] = count
            logger.debug(f"会话 '{session_id}' 消息计数: {count}/{self.rag_archive_threshold}")

            if count >= self.rag_archive_threshold:
                logger.info(f"会话 '{session_id}' 达到归档阈值，准备归档...")
                # 使用 create_task 在后台执行归档，避免阻塞当前请求
                asyncio.create_task(helpers.archive_conversation(self, event))
                self.session_message_counts[session_id] = 0
                logger.info(f"会话 '{session_id}' 消息计数器已重置。")

    async def terminate(self):
        """
        插件卸载时清理资源。
        """
        logger.info("RAGFlow 适配器插件已终止。")
