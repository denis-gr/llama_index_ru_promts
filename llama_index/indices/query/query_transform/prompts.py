"""Query transform prompts."""


from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

# deprecated, kept for backwards compatibility
"""Decompose prompt for query transformation.

PromptTemplate to "decompose" a query into another query
given the existing context.

Required template variables: `context_str`, `query_str`
"""
DecomposeQueryTransformPrompt = PromptTemplate

"""Step Decompose prompt for query transformation.

PromptTemplate to "decompose" a query into another query
given the existing context + previous reasoning (the previous steps).

Required template variables: `context_str`, `query_str`, `prev_reasoning`
"""
StepDecomposeQueryTransformPrompt = PromptTemplate

"""Image output prompt for query transformation.

PromptTemplate to add instructions for formatting image output.

Required template variables: `query_str`, `image_width`
"""
ImageOutputQueryTransformPrompt = PromptTemplate


DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below. \n"
    "Given the context, return a new question that can be answered from "
    "the context. The question can be the same as the original question, "
    "or a new question that represents a subcomponent of the overall question.\n"
    "As an example: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: What is the current population of the city in which Paul Graham found "
    "his first company, Viaweb?\n"
    "Knowledge source context: Provides information about Paul Graham's "
    "professional career, including the startups he's founded. "
    "New question: In which city did Paul Graham found his first company, Viaweb? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "New question: "
)

DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
"Исходный вопрос следующий: {query_str}\n"
    "У нас есть возможность ответить на некоторые или все вопросы из "
    "источник знаний."
    "Контекстная информация для источника знаний приведена ниже. \n"
    "Учитывая контекст, верните новый вопрос, на который можно ответить из "
    "- контекст. Вопрос может быть таким же, как и исходный вопрос, "
    "или новый вопрос, который представляет собой подкомпонент общего вопроса.\n"
    "В качестве примера: "
    "\n\n"
    'Вопрос: Сколько титулов "Большого шлема" у победителя Чемпионата Австралии 2020 года'
    "Открыть есть?\n"
    "Контекст источника знаний: предоставляет информацию о победителях конкурса 2020 года "
    "Открытый чемпионат Австралии"
    "Новый вопрос: кто стал победителем Открытого чемпионата Австралии 2020 года?"
    "\n\n"
    "Вопрос: Каково нынешнее население города, в котором Пол Грэм нашел"
    "его первая компания, Viaweb?\n"
    "Контекст источника знаний: предоставляет информацию о работе Пола Грэхема "
    "профессиональная карьера, включая стартапы, которые он основал."
    "Новый вопрос: в каком городе Пол Грэм основал свою первую компанию Viaweb?"
    "\n\n"
    "Вопрос: {query_str}\n"
    "Контекст источника знаний: {context_str}\n"
    "Новый вопрос: "
)

DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT = PromptTemplate(
    DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL, prompt_type=PromptType.DECOMPOSE
)


DEFAULT_IMAGE_OUTPUT_TMPL = (
    "{query_str}"
    "Show any image with a HTML <img/> tag with {image_width}."
    'e.g., <image src="data/img.jpg" width="{image_width}" />.'
)

DEFAULT_IMAGE_OUTPUT_TMPL = (
"{query_str}"
    "Показывать любое изображение с тегом HTML <img/> с {image_width}."
    'например, <image src="data/img.jpg " ширина="{image_width}" />.'
)
DEFAULT_IMAGE_OUTPUT_PROMPT = PromptTemplate(DEFAULT_IMAGE_OUTPUT_TMPL)


DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below, as "
    "well as previous reasoning steps.\n"
    "Given the context and previous reasoning, return a question that can "
    "be answered from "
    "the context. This question can be the same as the original question, "
    "or this question can represent a subcomponent of the overall question."
    "It should not be irrelevant to the original question.\n"
    "If we cannot extract more information from the context, provide 'None' "
    "as the answer. "
    "Some examples are given below: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides names of the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None\n"
    "Next question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: Who was the winner of the 2020 Australian Open?\n"
    "Knowledge source context: Provides names of the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None.\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: None"
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open - includes biographical information for each winner\n"
    "Previous reasoning:\n"
    "- Who was the winner of the 2020 Australian Open? \n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: How many Grand Slam titles does Novak Djokovic have? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "Previous reasoning: {prev_reasoning}\n"
    "New question: "
)

DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
"Исходный вопрос следующий: {query_str}\n"
    "У нас есть возможность ответить на некоторые или все вопросы из "
    "источник знаний."
    "Контекстная информация для источника знаний представлена ниже, как "
    "так же, как и предыдущие шаги рассуждения.\n"
    "Учитывая контекст и предыдущие рассуждения, задайте вопрос, который может "
    "получить ответ от "
    "- контекст. Этот вопрос может быть таким же, как и исходный вопрос, "
    "или этот вопрос может представлять собой подкомпонент общего вопроса."
    "Это не должно иметь отношения к первоначальному вопросу.\n"
    "Если мы не можем извлечь больше информации из контекста, укажите НЕТ"
    "в качестве ответаю"
    "Ниже приведены некоторые примеры: "
    "\n\n"
    'Вопрос: Сколько титулов "Большого шлема" у победителя Чемпионата Австралии 2020 года'
    "Открыть есть?\n"
    "Контекст источника знаний: приведены имена победителей конкурса 2020 года "
    "Открытый чемпионат Австралии"
    "Предыдущее рассуждение: нет\n"
    "Следующий вопрос: кто стал победителем Открытого чемпионата Австралии 2020 года?"
    "\n\n"
    "Вопрос: Кто стал победителем Открытого чемпионата Австралии 2020 года?\n"
    "Контекст источника знаний: приведены имена победителей конкурса 2020 года "
    "Открытый чемпионат Австралии"
    "Предыдущее рассуждение: Нет.\n"
    "Новый вопрос: кто стал победителем Открытого чемпионата Австралии 2020 года?"
    "\n\n"
    'Вопрос: Сколько титулов "Большого шлема" у победителя Чемпионата Австралии 2020 года'
    "Открыть есть?\n"
    "Контекст источника знаний: предоставляет информацию о победителях конкурса 2020 года "
    "Открытый чемпионат Австралии"
    "Предыдущее рассуждение:\n"
    "- Кто стал победителем Открытого чемпионата Австралии 2020 года? \n"
    "- Победителем Открытого чемпионата Австралии 2020 года стал Новак Джокович.\n"
    "Новый вопрос: нет"
    "\n\n"
    'Вопрос: Сколько титулов "Большого шлема" у победителя Чемпионата Австралии 2020 года'
    "Открыть есть?\n"
    "Контекст источника знаний: предоставляет информацию о победителях конкурса 2020 года "
    "Открытый чемпионат Австралии - включает биографическую информацию о каждом победителе\n"
    "Предыдущее рассуждение:\n"
    "- Кто стал победителем Открытого чемпионата Австралии 2020 года? \n"
    "- Победителем Открытого чемпионата Австралии 2020 года стал Новак Джокович.\n"
    'Новый вопрос: сколько титулов "Большого шлема" у Новака Джоковича?'
    "\n\n"
    "Вопрос: {query_str}\n"
    "Контекст источника знаний: {context_str}\n"
    "Предыдущее рассуждение: {prev_reasoning}\n"
    "Новый вопрос: "
)

DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_PROMPT = PromptTemplate(
    DEFAULT_STEP_DECOMPOSE_QUERY_TRANSFORM_TMPL
)
