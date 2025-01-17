from typing import Any, List, Optional, Sequence

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.schema import QueryBundle
from llama_index.prompts import PromptTemplate
from llama_index.prompts.base import BasePromptTemplate
from llama_index.response.schema import RESPONSE_TYPE
from llama_index.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.schema import NodeWithScore, TextNode
from llama_index.text_splitter import get_default_text_splitter
from llama_index.text_splitter.types import TextSplitter

CITATION_QA_TEMPLATE = PromptTemplate(

    "Пожалуйста, дайте ответ, основанный исключительно на предоставленных источниках."
    "Когда ссылаешься на информацию из источника",
    "процитируйте соответствующий источник (источники), используя их соответствующие номера."
    "Каждый ответ должен включать по крайней мере одну цитату из источника."
    "Цитируйте источник только тогда, когда вы явно ссылаетесь на него. "
"Если ни один из источников не является полезным, вы должны указать это. "
"Например:\n"
    "Источник 1:\n"
    "Небо красное вечером и голубое утром.\n"
    "Источник 2:\n"
    "Вода мокрая, когда небо красное.\n"
    "Запрос: Когда вода становится влажной?\n"
    "Ответ: Вода будет влажной, когда небо станет красным [2]",
    "который происходит вечером [1].\n"
    "- Теперь твоя очередь. Ниже приведены несколько пронумерованных источников информации:"
    "\n-------\n"
    "{context_str}"
    "\n-------\n"
    "Запрос: {query_str}\n"
    "Ответ: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(

    "Пожалуйста, дайте ответ, основанный исключительно на предоставленных источниках."
"Когда ссылаешься на информацию из источника",
"процитируйте соответствующий источник (источники), используя их соответствующие номера."
"Каждый ответ должен включать по крайней мере одну цитату из источника."
"Цитируйте источник только тогда, когда вы явно ссылаетесь на него. "
"Если ни один из источников не является полезным, вы должны указать это. "
"Например:\n"
"Источник 1:\n"
"Небо красное вечером и голубое утром.\n"
"Источник 2:\n"
"Вода мокрая, когда небо красное.\n"
"Запрос: Когда вода становится влажной?\n"
"Ответ: Вода будет влажной, когда небо станет красным [2]",
"который происходит вечером [1].\n"
"Теперь твоя очередь."
"Мы предоставили существующий ответ: {existing_answer}"
"Ниже приведены несколько пронумерованных источников информации."
"Используйте их, чтобы уточнить существующий ответ."
"Если предоставленные источники не помогут, вы повторите существующий ответ."
"Начинаем рафинирование!"
"\n-------\n"
"{context_msg}"
"\n-------\n"
"Запрос: {query_str}\n"
"Ответ: "
)

DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 20


class CitationQueryEngine(BaseQueryEngine):
    """Citation query engine.

    Args:
        retriever (BaseRetriever): A retriever object.
        response_synthesizer (Optional[BaseSynthesizer]):
            A BaseSynthesizer object.
        citation_chunk_size (int):
            Size of citation chunks, default=512. Useful for controlling
            granularity of sources.
        citation_chunk_overlap (int): Overlap of citation nodes, default=20.
        text_splitter (Optional[TextSplitterType]):
            A text splitter for creating citation source nodes. Default is
            a SentenceSplitter.
        callback_manager (Optional[CallbackManager]): A callback manager.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self.text_splitter = text_splitter or get_default_text_splitter(
            chunk_size=citation_chunk_size, chunk_overlap=citation_chunk_overlap
        )
        self._retriever = retriever
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            service_context=retriever.get_service_context(),
            callback_manager=callback_manager,
        )
        self._node_postprocessors = node_postprocessors or []

        super().__init__(callback_manager)

    @classmethod
    def from_args(
        cls,
        index: BaseGPTIndex,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        citation_chunk_size: int = DEFAULT_CITATION_CHUNK_SIZE,
        citation_chunk_overlap: int = DEFAULT_CITATION_CHUNK_OVERLAP,
        text_splitter: Optional[TextSplitter] = None,
        citation_qa_template: BasePromptTemplate = CITATION_QA_TEMPLATE,
        citation_refine_template: BasePromptTemplate = CITATION_REFINE_TEMPLATE,
        retriever: Optional[BaseRetriever] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        # response synthesizer args
        response_mode: ResponseMode = ResponseMode.COMPACT,
        use_async: bool = False,
        streaming: bool = False,
        # class-specific args
        **kwargs: Any,
    ) -> "CitationQueryEngine":
        """Initialize a CitationQueryEngine object."

        Args:
            index: (BastGPTIndex): index to use for querying
            citation_chunk_size (int):
                Size of citation chunks, default=512. Useful for controlling
                granularity of sources.
            citation_chunk_overlap (int): Overlap of citation nodes, default=20.
            text_splitter (Optional[TextSplitter]):
                A text splitter for creating citation source nodes. Default is
                a SentenceSplitter.
            citation_qa_template (BasePromptTemplate): Template for initial citation QA
            citation_refine_template (BasePromptTemplate):
                Template for citation refinement.
            retriever (BaseRetriever): A retriever object.
            service_context (Optional[ServiceContext]): A ServiceContext object.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of
                node postprocessors.
            verbose (bool): Whether to print out debug info.
            response_mode (ResponseMode): A ResponseMode object.
            use_async (bool): Whether to use async.
            streaming (bool): Whether to use streaming.
            optimizer (Optional[BaseTokenUsageOptimizer]): A BaseTokenUsageOptimizer
                object.

        """
        retriever = retriever or index.as_retriever(**kwargs)

        response_synthesizer = response_synthesizer or get_response_synthesizer(
            service_context=index.service_context,
            text_qa_template=citation_qa_template,
            refine_template=citation_refine_template,
            response_mode=response_mode,
            use_async=use_async,
            streaming=streaming,
        )

        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            callback_manager=index.service_context.callback_manager,
            citation_chunk_size=citation_chunk_size,
            citation_chunk_overlap=citation_chunk_overlap,
            text_splitter=text_splitter,
            node_postprocessors=node_postprocessors,
        )

    def _create_citation_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Modify retrieved nodes to be granular sources."""
        new_nodes: List[NodeWithScore] = []
        for node in nodes:
            text_chunks = self.text_splitter.split_text(node.node.get_content())

            for text_chunk in text_chunks:
                text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"

                new_nodes.append(
                    NodeWithScore(
                        node=TextNode(
                            text=text,
                            metadata=node.node.metadata or {},
                            relationships=node.node.relationships or {},
                        ),
                        score=node.score,
                    )
                )
        return new_nodes

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        nodes = self._retriever.retrieve(query_bundle)

        for postprocessor in self._node_postprocessors:
            nodes = postprocessor.postprocess_nodes(nodes)

        return nodes

    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever object."""
        return self._retriever

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        nodes = self._create_citation_nodes(nodes)
        response = self._response_synthesizer.synthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )
        return response

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        nodes = self._create_citation_nodes(nodes)
        return await self._response_synthesizer.asynthesize(
            query=query_bundle,
            nodes=nodes,
            additional_source_nodes=additional_source_nodes,
        )

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)
                nodes = self._create_citation_nodes(nodes)

                retrieve_event.on_end(payload={EventPayload.NODES: nodes})

            response = self._response_synthesizer.synthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                nodes = self.retrieve(query_bundle)
                nodes = self._create_citation_nodes(nodes)

                retrieve_event.on_end(payload={EventPayload.NODES: nodes})

            response = await self._response_synthesizer.asynthesize(
                query=query_bundle,
                nodes=nodes,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response
