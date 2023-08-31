"""Evaluating the responses from an index."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from llama_index.indices.base import ServiceContext
from llama_index.indices.list.base import ListIndex
from llama_index.prompts import PromptTemplate
from llama_index.schema import Document
from llama_index.response.schema import Response


@dataclass
class Evaluation:
    query: str  # The query
    response: Response  # The response
    passing: bool = False  # True if the response is correct, False otherwise
    feedback: str = ""  # Feedback for the response


class BaseEvaluator(ABC):
    def __init__(self, service_context: Optional[ServiceContext] = None) -> None:
        """Base class for evaluating responses"""
        self.service_context = service_context or ServiceContext.from_defaults()

    @abstractmethod
    def evaluate_response(self, query: str, response: Response) -> Evaluation:
        """Evaluate the response for a query and return an Evaluation."""
        raise NotImplementedError


DEFAULT_EVAL_PROMPT = (
    "Пожалуйста, сообщите, соответствует ли данная информация "
    "поддерживается контекстом.\n"
    "Вам нужно ответить либо ДА, либо НЕТ.\n"
    "Ответьте ДА, если какой-либо контекст подтверждает эту информацию, даже "
    "если большая часть контекста не имеет отношения к делу."
    "Ниже приведены некоторые примеры. \n\n"
    "Информация: Яблочный пирог, как правило, с двойной корочкой.\n"
    "Контекст: Яблочный пирог - это фруктовый пирог, в котором основная начинка "
    "- ингредиент - яблоки. \n"
    "Яблочный пирог часто подают со взбитыми сливками, мороженым"
    "(яблочный пирог а-ля мод), заварной крем или сыр чеддер.\n"
    "Обычно это блюдо с двойной корочкой, с тестом сверху"
    "и под начинкой; верхняя корочка может быть твердой или "
    "решетчатый (сплетенный из поперечных полос).\n"
    "Ответ: ДА\n"
    "Информация: Яблочные пироги невкусные.\n"
    "Контекст: Яблочный пирог - это фруктовый пирог, в котором основная начинка "
    "- ингредиент - яблоки. \n"
    "Яблочный пирог часто подают со взбитыми сливками, мороженым"
    "(\"яблочный пирог а-ля мод\"), заварной крем или сыр чеддер.\n"
    "Обычно это блюдо с двойной корочкой, с тестом сверху"
    "и под начинкой; верхняя корочка может быть твердой или "
    "решетчатый (сплетенный из поперечных полос).\n"
    "Ответ: НЕТ\n"
    "Контекст в <>: <{context_str}>\n"
    "Информация в []: [{query_str}]\n"
    "Твой ответ: "
)

DEFAULT_REFINE_PROMPT = (
    "Мы хотим понять, присутствует ли следующая информация "
    "в контекстной информации: {query_str}\n"
    "Мы предоставили существующий ответ ДА/НЕТ: {existing_answer}\n"
    "У нас есть возможность уточнить существующий ответ "
    "(только при необходимости) с дополнительным контекстом ниже.\n"
    "------------\ n"
    "{context_msg}\n"
    "------------\ n"
    "Если существующий ответ уже был ДА, все равно отвечайте ДА."
    "Если информация присутствует в новом контексте, ответьте ДА."
    "В противном случае отвечайте НЕТ.\n"
)

QUERY_RESPONSE_EVAL_PROMPT = (
    "Ваша задача состоит в том, чтобы оценить, соответствует ли ответ на запрос \
    соответствует предоставленной контекстной информации.\n"
    "У вас есть два варианта ответа. Либо ДА/НЕТ.\n"
    "Ответ - ДА, если ответ на запрос \
    соответствует контекстной информации, в противном случае НЕТ.\n"
    "Запрос и ответ: \n {query_str}\n"
    "Контекст: \n {context_str}\n"
    "Ответ: "
)

QUERY_RESPONSE_REFINE_PROMPT = (
"Мы хотим понять, является ли следующий запрос и ответ"
    "в соответствии с контекстной информацией: \n {query_str}\n"
    "Мы предоставили существующий ответ ДА/НЕТ: \n {existing_answer}\n"
    "У нас есть возможность уточнить существующий ответ "
    "(только при необходимости) с дополнительным контекстом ниже.\n"
    "------------\ n"
    "{context_str}\n"
    "------------\ n"
    "Если существующий ответ уже был ДА, все равно отвечайте ДА."
    "Если информация присутствует в новом контексте, ответьте ДА."
    "В противном случае отвечайте НЕТ.\n"
)


class ResponseEvaluator:
    """Evaluate based on response from indices.

    NOTE: this is a beta feature, subject to change!

    Args:
        service_context (Optional[ServiceContext]): ServiceContext object

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        raise_error: bool = False,
    ) -> None:
        """Init params."""
        self.service_context = service_context or ServiceContext.from_defaults()
        self.raise_error = raise_error

    def get_context(self, response: Response) -> List[Document]:
        """Get context information from given Response object using source nodes.

        Args:
            response (Response): Response object from an index based on the query.

        Returns:
            List of Documents of source nodes information as context information.
        """

        context = []

        for context_info in response.source_nodes:
            context.append(Document(text=context_info.node.get_content()))

        return context

    def evaluate(self, response: Response) -> str:
        """Evaluate the response from an index.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            Yes -> If answer, context information are matching \
                    or If Query, answer and context information are matching.
            No -> If answer, context information are not matching \
                    or If Query, answer and context information are not matching.
        """
        answer = str(response)

        context = self.get_context(response)
        index = ListIndex.from_documents(context, service_context=self.service_context)
        response_txt = ""

        EVAL_PROMPT_TMPL = PromptTemplate(DEFAULT_EVAL_PROMPT)
        REFINE_PROMPT_TMPL = PromptTemplate(DEFAULT_REFINE_PROMPT)

        query_engine = index.as_query_engine(
            text_qa_template=EVAL_PROMPT_TMPL,
            refine_template=REFINE_PROMPT_TMPL,
        )
        response_obj = query_engine.query(answer)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            response_txt = "YES"
        else:
            response_txt = "NO"
            if self.raise_error:
                raise ValueError("The response is invalid")

        return response_txt

    def evaluate_source_nodes(self, response: Response) -> List[str]:
        """Function to evaluate if each source node contains the answer \
            by comparing the response, and context information (source node).

        Args:
            response: Response object from an index based on the query.
        Returns:
            List of Yes/ No which can be used to know which source node contains \
                answer.
            Yes -> If response and context information are matching.
            No -> If response and context information are not matching.
        """
        answer = str(response)

        context_list = self.get_context(response)

        response_texts = []

        for context in context_list:
            index = ListIndex.from_documents(
                [context], service_context=self.service_context
            )
            response_txt = ""

            EVAL_PROMPT_TMPL = PromptTemplate(DEFAULT_EVAL_PROMPT)
            REFINE_PROMPT_TMPL = PromptTemplate(DEFAULT_REFINE_PROMPT)

            query_engine = index.as_query_engine(
                text_qa_template=EVAL_PROMPT_TMPL,
                refine_template=REFINE_PROMPT_TMPL,
            )
            response_obj = query_engine.query(answer)
            raw_response_txt = str(response_obj)

            if "yes" in raw_response_txt.lower():
                response_txt = "YES"
            else:
                response_txt = "NO"
                if self.raise_error:
                    raise ValueError("The response is invalid")

            response_texts.append(response_txt)

        return response_texts


class QueryResponseEvaluator(BaseEvaluator):
    """Evaluate based on query and response from indices.

    NOTE: this is a beta feature, subject to change!

    Args:
        service_context (Optional[ServiceContext]): ServiceContext object

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        raise_error: bool = False,
    ) -> None:
        """Init params."""
        super().__init__(service_context)
        self.raise_error = raise_error

    def get_context(self, response: Response) -> List[Document]:
        """Get context information from given Response object using source nodes.

        Args:
            response (Response): Response object from an index based on the query.

        Returns:
            List of Documents of source nodes information as context information.
        """

        context = []

        for context_info in response.source_nodes:
            context.append(Document(text=context_info.node.get_content()))

        return context

    def evaluate(self, query: str, response: Response) -> str:
        """Evaluate the response from an index.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            Yes -> If answer, context information are matching \
                    or If Query, answer and context information are matching.
            No -> If answer, context information are not matching \
                    or If Query, answer and context information are not matching.
        """
        return self.evaluate_response(query, response).feedback

    def evaluate_response(self, query: str, response: Response) -> Evaluation:
        """Evaluate the response from an index.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            Evaluation object with passing boolean and feedback "YES" or "NO".
        """
        answer = str(response)

        context = self.get_context(response)
        index = ListIndex.from_documents(context, service_context=self.service_context)

        QUERY_RESPONSE_EVAL_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_EVAL_PROMPT)
        QUERY_RESPONSE_REFINE_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_REFINE_PROMPT)

        query_response = f"Question: {query}\nResponse: {answer}"

        query_engine = index.as_query_engine(
            text_qa_template=QUERY_RESPONSE_EVAL_PROMPT_TMPL,
            refine_template=QUERY_RESPONSE_REFINE_PROMPT_TMPL,
        )
        response_obj = query_engine.query(query_response)

        raw_response_txt = str(response_obj)

        if "yes" in raw_response_txt.lower():
            return Evaluation(query, response, True, "YES")
        else:
            if self.raise_error:
                raise ValueError("The response is invalid")
            return Evaluation(query, response, False, "NO")

    def evaluate_source_nodes(self, query: str, response: Response) -> List[str]:
        """Function to evaluate if each source node contains the answer \
            to a given query by comparing the query, response, \
                and context information.

        Args:
            query: Query for which response is generated from index.
            response: Response object from an index based on the query.
        Returns:
            List of Yes/ No which can be used to know which source node contains \
                answer.
            Yes -> If answer, context information are matching \
                    or If Query, answer and context information are matching \
                        for a source node.
            No -> If answer, context information are not matching \
                    or If Query, answer and context information are not matching \
                        for a source node.
        """
        answer = str(response)

        context_list = self.get_context(response)

        response_texts = []

        for context in context_list:
            index = ListIndex.from_documents(
                [context], service_context=self.service_context
            )
            response_txt = ""

            QUERY_RESPONSE_EVAL_PROMPT_TMPL = PromptTemplate(QUERY_RESPONSE_EVAL_PROMPT)
            QUERY_RESPONSE_REFINE_PROMPT_TMPL = PromptTemplate(
                QUERY_RESPONSE_REFINE_PROMPT
            )

            query_response = f"Question: {query}\nResponse: {answer}"

            query_engine = index.as_query_engine(
                text_qa_template=QUERY_RESPONSE_EVAL_PROMPT_TMPL,
                refine_template=QUERY_RESPONSE_REFINE_PROMPT_TMPL,
            )
            response_obj = query_engine.query(query_response)
            raw_response_txt = str(response_obj)

            if "yes" in raw_response_txt.lower():
                response_txt = "YES"
            else:
                response_txt = "NO"
                if self.raise_error:
                    raise ValueError("The response is invalid")

            response_texts.append(response_txt)

        return response_texts
