"""Prompts for ChatGPT."""

from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate

# text qa prompt
TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "Вы являетесь экспертом в системе вопросов и ответов, которой доверяют во всем мире.\n"
        "Всегда отвечайте на запрос, используя предоставленную контекстную информацию",
        "и без предварительного знания.\n"
        "Некоторые правила, которым следует следовать:\n"
        "1. Никогда не ссылайтесь напрямую на данный контекст в своем ответе.\n"
        "2. Избегайте утверждений типа 'Исходя из контекста, ...' или "
        "'Контекстная информация...' или что-нибудь в этом роде"
        "эти линии."
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Контекстная информация приведена ниже.\n"
            "---------------------\ n"
            "{context_str}\n"
            "---------------------\ n"
            "Учитывая контекстную информацию, а не предварительные знания",
            "ответьте на запрос.\n"
            "Запрос: {query_str}\n"
            "Ответ: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

# Tree Summarize
TREE_SUMMARIZE_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Ниже приведена контекстная информация из нескольких источников.\n"
            "---------------------\ n"
            "{context_str}\n"
            "---------------------\ n"
            "Учитывая информацию из нескольких источников, а не предварительные знания",
            "ответьте на запрос.\n"
            "Запрос: {query_str}\n"
            "Ответ: "
        ),
        role=MessageRole.USER,
    ),
]

CHAT_TREE_SUMMARIZE_PROMPT = ChatPromptTemplate(
    message_templates=TREE_SUMMARIZE_PROMPT_TMPL_MSGS
)


# Refine Prompt
CHAT_REFINE_PROMPT_TMPL_MSGS = [
    ChatMessage(
        content=(
            "Вы являетесь экспертной системой вопросов и ответов, которая работает строго в двух режимах"
            "при уточнении существующих ответов:\n"
            "1. ** Перепишите** оригинальный ответ, используя новый контекст.\n"
            "2. ** Повторите ** первоначальный ответ, если новый контекст бесполезен.\n"
            "Никогда не ссылайтесь непосредственно на исходный ответ или контекст в своем ответе.\n"
            "Если сомневаетесь, просто повторите первоначальный ответ."
            "Новый контекст: {context_msg}\n"
            "Запрос: {query_str}\n"
            "Оригинальный ответ: {existing_answer}\n"
            "Новый ответ: "
        ),
        role=MessageRole.USER,
    )
]


CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)


# Table Context Refine Prompt
CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS = [
    ChatMessage(content="{query_str}", role=MessageRole.USER),
    ChatMessage(content="{existing_answer}", role=MessageRole.ASSISTANT),
    ChatMessage(
        content=(
            "Мы предоставили схему таблицы ниже."
            "---------------------\ n"
            "{схема}\n"
            "---------------------\ n"
            "Мы также предоставили некоторую контекстную информацию ниже."
            "{context_msg}\n"
            "---------------------\ n"
            "Учитывая контекстную информацию и схему таблицы",
            "доработайте первоначальный ответ, чтобы он стал лучше "
            " отвечай на вопрос."
            "Если контекст бесполезен, верните исходный ответ."
        ),
        role=MessageRole.USER,
    ),
]
CHAT_REFINE_TABLE_CONTEXT_PROMPT = ChatPromptTemplate(
    message_templates=CHAT_REFINE_TABLE_CONTEXT_TMPL_MSGS
)
