import asyncio
from contextlib import suppress

from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.processors.lang_adapter_router import LangAdapterRouter, detect_language

ADAPTERS = {'fr': '/models/fr_lora', 'ja': '/models/ja_lora', 'qwen-refit': '/models/refit'}


class _FakePromptProcessor:
    """Record the adapter_name the engine resolved for a request."""

    def __init__(self):
        self.seen = []

    async def get_prompt_input(self, **kwargs):
        self.seen.append(kwargs.get('adapter_name'))
        raise asyncio.CancelledError


class _FakeRequestLogger:

    def log_prompt(self, *args, **kwargs):
        pass


def _make_engine(adapters=ADAPTERS):
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SessionManager()
    engine.prompt_processor = _FakePromptProcessor()
    engine.request_logger = _FakeRequestLogger()
    engine.lang_adapter_router = LangAdapterRouter(adapters)
    return engine


async def _submit(engine, prompt):
    """Drive generate() until prompt preprocessing resolves an adapter."""
    generator = engine.generate(prompt, 260606)
    with suppress(asyncio.CancelledError):
        await generator.__anext__()


def test_detect_language_scripts():
    assert detect_language('Le chat est sur la table et il ne veut pas sortir') == 'fr'
    assert detect_language('こんにちは、今日はいい天気ですね。') == 'ja'
    assert detect_language('这是一段中文文本，用来测试语言检测。') == 'zh'
    assert detect_language('안녕하세요, 오늘 날씨가 좋네요.') == 'ko'
    assert detect_language('Привет, как дела? Это не сложно и очень хорошо') == 'ru'


def test_detect_language_requires_signal():
    assert detect_language('') is None
    assert detect_language('hi') is None
    assert detect_language('12345678 !!!') is None


def test_router_only_routes_language_code_adapters():
    router = LangAdapterRouter(ADAPTERS)
    assert sorted(router.routed_adapters()) == ['fr', 'ja']
    assert router.route('Le chat et la maison sont grands') == 'fr'
    # English and unrouted languages stay on the base model.
    assert router.route('The quick brown fox jumps over the lazy dog') is None
    assert router.route('Der Hund ist gross und laeuft schnell') is None
    assert LangAdapterRouter(None).enabled is False


def test_engine_routes_unnamed_request_to_language_adapter():
    engine = _make_engine()
    asyncio.run(_submit(engine, 'Bonjour, quel est le sens de la vie ?'))
    assert engine.prompt_processor.seen == ['fr']


def test_engine_routes_multimodal_messages_by_text_content():
    engine = _make_engine()
    messages = [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'https://example.com/a.png'}},
        {'type': 'text', 'text': 'この画像について説明してください。これはテストです。'},
    ]}]
    asyncio.run(_submit(engine, messages))
    assert engine.prompt_processor.seen == ['ja']


def test_engine_keeps_explicit_adapter_name():
    engine = _make_engine()
    generator = engine.generate('Bonjour et merci', 260607, adapter_name='qwen-refit')
    with suppress(asyncio.CancelledError):
        asyncio.run(generator.__anext__())
    assert engine.prompt_processor.seen == ['qwen-refit']


def test_engine_without_adapters_leaves_adapter_unset():
    engine = _make_engine(adapters=None)
    asyncio.run(_submit(engine, 'Bonjour et merci pour la reponse'))
    assert engine.prompt_processor.seen == [None]


def test_bare_engine_has_inert_router_default():
    # Instances built without __init__ (tests, engine wrappers) must fall back
    # to the class-level disabled router instead of raising AttributeError.
    engine = AsyncEngine.__new__(AsyncEngine)
    assert engine.lang_adapter_router.route('Bonjour et merci pour la reponse') is None


def test_routing_prompt_text_ignores_non_text_content():
    text = AsyncEngine._routing_prompt_text([{
        'role': 'user',
        'content': [{'type': 'image_url', 'image_url': {'url': 'x'}},
                    {'type': 'text', 'text': 'この画像'}],
    }])
    assert text == 'この画像'
    # OpenAI messages also allow content to be a plain string.
    assert AsyncEngine._routing_prompt_text([{'role': 'user', 'content': 'bonjour'}]) == 'bonjour'
    assert AsyncEngine._routing_prompt_text(None) is None
