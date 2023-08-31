"""Answer inserter."""

from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.query_engine.flare.schema import QueryTask
from llama_index.prompts.base import BasePromptTemplate, PromptTemplate
from llama_index.indices.service_context import ServiceContext


class BaseLookaheadAnswerInserter(ABC):
    """Lookahead answer inserter.

    These are responsible for insert answers into a lookahead answer template.

    E.g.
    lookahead answer: Red is for [Search(What is the meaning of Ghana's
        flag being red?)], green for forests, and gold for mineral wealth.
    query: What is the meaning of Ghana's flag being red?
    query answer: "the blood of those who died in the country's struggle
        for independence"
    final answer: Red is for the blood of those who died in the country's
        struggle for independence, green for forests, and gold for mineral wealth.

    """

    @abstractmethod
    def insert(
        self,
        response: str,
        query_tasks: List[QueryTask],
        answers: List[str],
        prev_response: Optional[str] = None,
    ) -> str:
        """Insert answers into response."""


DEFAULT_ANSWER_INSERT_PROMPT_TMPL = """
Существующий "предварительный ответ" приведен ниже. Ответ lookahead
содержит теги `[Поиск(запрос)]`. Некоторые запросы были выполнены и
получен ответ. Вопросы и ответы также приведены ниже.
Также предыдущий ответ (ответ перед предварительным ответом)
приведен ниже.
Учитывая шаблон предварительного просмотра, предыдущий ответ, а также запросы и ответы на них,
пожалуйста, "заполните" шаблон предварительного просмотра соответствующими ответами.

ПРИМЕЧАНИЕ: Пожалуйста, убедитесь, что окончательный ответ грамматически соответствует
предыдущий ответ + шаблон предварительного просмотра. Например, если предыдущий
ответ был "Население Нью-Йорка составляет "
, а шаблон предварительного просмотра - "[Поиск (каково население Нью-Йорка?)]", то
окончательный ответ должен быть "8,4 миллиона".

ПРИМЕЧАНИЕ: шаблон предварительного просмотра может быть неполным предложением и
содержать конечные/ ведущие запятые и т.д. Пожалуйста, сохраните исходное
форматирование шаблона предварительного просмотра, если это возможно.

записка: 

ПРИМЕЧАНИЕ: исключением из приведенного выше правила является случай, когда ответ на запрос
это эквивалентно "я не знаю" или "у меня нет ответа". В этом случае
измените шаблон предварительного просмотра, чтобы указать, что ответ неизвестен.

ПРИМЕЧАНИЕ: шаблон lookahead может содержать несколько тегов `[Поиск(запрос)]`
    и только подмножество этих запросов было выполнено.
    Не заменяйте теги `[Поиск(запрос)]`, которые не были выполнены.

Предыдущий ответ:


Предварительный шаблон:
Красный означает [Поиск(что означает название Ганы \
    флаг красный?)], зеленый - для лесов, а золотой - для полезных ископаемых.

Пары "Запрос-ответ":
Вопрос: Что означает то, что флаг Ганы красный?
Ответ: Красный цвет символизирует кровь тех, кто погиб в борьбе за страну \
    за независимость

Заполненные ответы:
Красный цвет символизирует кровь тех, кто погиб в борьбе страны за независимость, \
    зеленый цвет - для лесов, а золотой - для полезных ископаемых.

Предыдущий ответ:
Один из крупнейших городов мира

Шаблон поиска:
, город содержит население [Поиск(каково население \
    из Нью-Йорка?)]

Пары "Запрос-ответ":
Вопрос: Каково население Нью-Йорка?
Ответ: Население Нью-Йорка составляет 8,4 миллиона человек

Обобщенный ответ:
население города составляет 8,4 миллиона человек

Предыдущий ответ:
население города составляет 

Предварительный шаблон:
[Поиск (Каково население Нью-Йорка?)]

Пары "Запрос-ответ":
Вопрос: Каково население Нью-Йорка?
Ответ: Население Нью-Йорка составляет 8,4 миллиона человек

Синтезированный ответ:
8,4 миллиона

Предыдущий ответ:
{prev_response}

Предварительный шаблон:
{lookahead_response}

Пары "Запрос-ответ":
{query_answer_pairs}

Синтезированный ответ:
"""
DEFAULT_ANSWER_INSERT_PROMPT = PromptTemplate(DEFAULT_ANSWER_INSERT_PROMPT_TMPL)


class LLMLookaheadAnswerInserter(BaseLookaheadAnswerInserter):
    """LLM Lookahead answer inserter.

    Takes in a lookahead response and a list of query tasks, and the
        lookahead answers, and inserts the answers into the lookahead response.

    Args:
        service_context (ServiceContext): Service context.

    """

    def __init__(
        self,
        service_context: Optional[ServiceContext] = None,
        answer_insert_prompt: Optional[BasePromptTemplate] = None,
    ) -> None:
        """Init params."""
        self._service_context = service_context or ServiceContext.from_defaults()
        self._answer_insert_prompt = (
            answer_insert_prompt or DEFAULT_ANSWER_INSERT_PROMPT
        )

    def insert(
        self,
        response: str,
        query_tasks: List[QueryTask],
        answers: List[str],
        prev_response: Optional[str] = None,
    ) -> str:
        """Insert answers into response."""
        prev_response = prev_response or ""

        query_answer_pairs = ""
        for query_task, answer in zip(query_tasks, answers):
            query_answer_pairs += f"Query: {query_task.query_str}\nAnswer: {answer}\n"

        response = self._service_context.llm_predictor.predict(
            self._answer_insert_prompt,
            lookahead_response=response,
            query_answer_pairs=query_answer_pairs,
            prev_response=prev_response,
        )
        return response


class DirectLookaheadAnswerInserter(BaseLookaheadAnswerInserter):
    """Direct lookahead answer inserter.

    Simple inserter module that directly inserts answers into
        the [Search(query)] tags in the lookahead response.

    Args:
        service_context (ServiceContext): Service context.

    """

    def insert(
        self,
        response: str,
        query_tasks: List[QueryTask],
        answers: List[str],
        prev_response: Optional[str] = None,
    ) -> str:
        """Insert answers into response."""
        for query_task, answer in zip(query_tasks, answers):
            response = (
                response[: query_task.start_idx]
                + answer
                + response[query_task.end_idx + 1 :]
            )
        return response
