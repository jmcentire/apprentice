"""
Tests for the Middleware Pipeline component.

Covers: MiddlewareContext, MiddlewareResponse, Middleware protocol,
MiddlewarePipeline (pre/post ordering, error resilience, state accumulation),
MiddlewareError, and from_config factory.
"""

import logging
import pytest
from unittest.mock import MagicMock

from apprentice.middleware import (
    Middleware,
    MiddlewareContext,
    MiddlewareError,
    MiddlewarePipeline,
    MiddlewareResponse,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def base_context():
    """A minimal valid MiddlewareContext."""
    return MiddlewareContext(
        request_id="req-001",
        task_name="summarize",
        input_data={"text": "hello world"},
    )


@pytest.fixture
def base_response():
    """A minimal valid MiddlewareResponse."""
    return MiddlewareResponse(
        output_data={"summary": "hello"},
    )


class PassthroughMiddleware:
    """Middleware that passes data through unchanged."""

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        return context

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        return response


class AnnotatingMiddleware:
    """Middleware that injects a marker into middleware_state so we can
    verify ordering and accumulation."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        pre_order = context.middleware_state.get("pre_order", [])
        new_state = {
            **context.middleware_state,
            "pre_order": pre_order + [self.tag],
            f"pre_{self.tag}": True,
        }
        return context.model_copy(update={"middleware_state": new_state})

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        post_order = response.middleware_state.get("post_order", [])
        new_state = {
            **response.middleware_state,
            "post_order": post_order + [self.tag],
            f"post_{self.tag}": True,
        }
        return response.model_copy(update={"middleware_state": new_state})


class FailingPreMiddleware:
    """Middleware whose pre_process always raises."""

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        raise RuntimeError("pre_process boom")

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        return response


class FailingPostMiddleware:
    """Middleware whose post_process always raises."""

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        return context

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        raise RuntimeError("post_process boom")


class TransformingMiddleware:
    """Middleware that modifies input_data (pre) and output_data (post)."""

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value

    def pre_process(self, context: MiddlewareContext) -> MiddlewareContext:
        new_input = {**context.input_data, self.key: self.value}
        return context.model_copy(update={"input_data": new_input})

    def post_process(
        self, context: MiddlewareContext, response: MiddlewareResponse
    ) -> MiddlewareResponse:
        new_output = {**response.output_data, self.key: self.value}
        return response.model_copy(update={"output_data": new_output})


# ===========================================================================
# MiddlewareContext Tests
# ===========================================================================


class TestMiddlewareContext:

    def test_creation_with_required_fields(self):
        ctx = MiddlewareContext(
            request_id="r1",
            task_name="classify",
            input_data={"text": "hi"},
        )
        assert ctx.request_id == "r1"
        assert ctx.task_name == "classify"
        assert ctx.input_data == {"text": "hi"}
        assert ctx.metadata == {}
        assert ctx.middleware_state == {}

    def test_creation_with_all_fields(self):
        ctx = MiddlewareContext(
            request_id="r2",
            task_name="translate",
            input_data={"text": "bonjour"},
            metadata={"lang": "fr"},
            middleware_state={"token_count": 5},
        )
        assert ctx.metadata == {"lang": "fr"}
        assert ctx.middleware_state == {"token_count": 5}

    def test_frozen_immutability(self, base_context):
        with pytest.raises((AttributeError, TypeError, Exception)):
            base_context.request_id = "changed"

    def test_frozen_immutability_input_data(self, base_context):
        with pytest.raises((AttributeError, TypeError, Exception)):
            base_context.input_data = {"new": "data"}

    def test_model_copy_produces_new_instance(self, base_context):
        updated = base_context.model_copy(
            update={"middleware_state": {"key": "val"}}
        )
        assert updated is not base_context
        assert updated.middleware_state == {"key": "val"}
        assert base_context.middleware_state == {}


# ===========================================================================
# MiddlewareResponse Tests
# ===========================================================================


class TestMiddlewareResponse:

    def test_creation_with_required_fields(self):
        resp = MiddlewareResponse(output_data={"result": 42})
        assert resp.output_data == {"result": 42}
        assert resp.metadata == {}
        assert resp.middleware_state == {}

    def test_creation_with_all_fields(self):
        resp = MiddlewareResponse(
            output_data={"result": 42},
            metadata={"model": "gpt-4"},
            middleware_state={"tokens_used": 100},
        )
        assert resp.metadata == {"model": "gpt-4"}
        assert resp.middleware_state == {"tokens_used": 100}

    def test_frozen_immutability(self, base_response):
        with pytest.raises((AttributeError, TypeError, Exception)):
            base_response.output_data = {"hacked": True}

    def test_model_copy_produces_new_instance(self, base_response):
        updated = base_response.model_copy(
            update={"middleware_state": {"flag": True}}
        )
        assert updated is not base_response
        assert updated.middleware_state == {"flag": True}
        assert base_response.middleware_state == {}


# ===========================================================================
# MiddlewareError Tests
# ===========================================================================


class TestMiddlewareError:

    def test_attributes(self):
        err = MiddlewareError("PiiTokenizer", "pre_process", "connection timeout")
        assert err.middleware_name == "PiiTokenizer"
        assert err.phase == "pre_process"
        assert err.reason == "connection timeout"

    def test_inherits_from_exception(self):
        err = MiddlewareError("X", "post_process", "boom")
        assert isinstance(err, Exception)

    def test_string_representation(self):
        err = MiddlewareError("MyMW", "pre_process", "bad input")
        assert "MyMW" in str(err)
        assert "pre_process" in str(err)
        assert "bad input" in str(err)


# ===========================================================================
# Middleware Protocol Tests
# ===========================================================================


class TestMiddlewareProtocol:

    def test_passthrough_satisfies_protocol(self):
        mw = PassthroughMiddleware()
        assert isinstance(mw, Middleware)

    def test_annotating_satisfies_protocol(self):
        mw = AnnotatingMiddleware("test")
        assert isinstance(mw, Middleware)

    def test_object_without_methods_fails_protocol(self):
        assert not isinstance("not a middleware", Middleware)
        assert not isinstance(42, Middleware)
        assert not isinstance({}, Middleware)

    def test_partial_implementation_fails_protocol(self):
        class OnlyPre:
            def pre_process(self, context):
                return context

        # Missing post_process, should not satisfy the protocol
        assert not isinstance(OnlyPre(), Middleware)


# ===========================================================================
# MiddlewarePipeline — Empty Pipeline (no-op)
# ===========================================================================


class TestEmptyPipeline:

    def test_execute_pre_passthrough(self, base_context):
        pipeline = MiddlewarePipeline([])
        result = pipeline.execute_pre(base_context)
        assert result == base_context

    def test_execute_post_passthrough(self, base_context, base_response):
        pipeline = MiddlewarePipeline([])
        result = pipeline.execute_post(base_context, base_response)
        assert result == base_response

    def test_none_middlewares_is_empty(self, base_context):
        pipeline = MiddlewarePipeline(None)
        result = pipeline.execute_pre(base_context)
        assert result == base_context

    def test_default_constructor_is_empty(self, base_context):
        pipeline = MiddlewarePipeline()
        result = pipeline.execute_pre(base_context)
        assert result == base_context


# ===========================================================================
# MiddlewarePipeline — Single Middleware
# ===========================================================================


class TestSingleMiddleware:

    def test_pre_process_runs(self, base_context):
        pipeline = MiddlewarePipeline([AnnotatingMiddleware("alpha")])
        result = pipeline.execute_pre(base_context)
        assert result.middleware_state.get("pre_alpha") is True

    def test_post_process_runs(self, base_context, base_response):
        pipeline = MiddlewarePipeline([AnnotatingMiddleware("alpha")])
        result = pipeline.execute_post(base_context, base_response)
        assert result.middleware_state.get("post_alpha") is True

    def test_passthrough_does_not_modify(self, base_context, base_response):
        pipeline = MiddlewarePipeline([PassthroughMiddleware()])
        pre_result = pipeline.execute_pre(base_context)
        assert pre_result.input_data == base_context.input_data

        post_result = pipeline.execute_post(base_context, base_response)
        assert post_result.output_data == base_response.output_data


# ===========================================================================
# MiddlewarePipeline — Multiple Middleware (ordering)
# ===========================================================================


class TestMultipleMiddlewareOrdering:

    def test_pre_process_runs_in_forward_order(self, base_context):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("first"),
            AnnotatingMiddleware("second"),
            AnnotatingMiddleware("third"),
        ])
        result = pipeline.execute_pre(base_context)
        assert result.middleware_state["pre_order"] == ["first", "second", "third"]

    def test_post_process_runs_in_reverse_order(self, base_context, base_response):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("first"),
            AnnotatingMiddleware("second"),
            AnnotatingMiddleware("third"),
        ])
        result = pipeline.execute_post(base_context, base_response)
        # Reverse order: third -> second -> first
        assert result.middleware_state["post_order"] == ["third", "second", "first"]

    def test_all_middleware_state_accumulated_pre(self, base_context):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("a"),
            AnnotatingMiddleware("b"),
        ])
        result = pipeline.execute_pre(base_context)
        assert result.middleware_state.get("pre_a") is True
        assert result.middleware_state.get("pre_b") is True

    def test_all_middleware_state_accumulated_post(self, base_context, base_response):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("a"),
            AnnotatingMiddleware("b"),
        ])
        result = pipeline.execute_post(base_context, base_response)
        assert result.middleware_state.get("post_a") is True
        assert result.middleware_state.get("post_b") is True

    def test_transforming_middleware_composes(self, base_context, base_response):
        pipeline = MiddlewarePipeline([
            TransformingMiddleware("injected_a", "val_a"),
            TransformingMiddleware("injected_b", "val_b"),
        ])
        pre_result = pipeline.execute_pre(base_context)
        assert pre_result.input_data["injected_a"] == "val_a"
        assert pre_result.input_data["injected_b"] == "val_b"

        post_result = pipeline.execute_post(base_context, base_response)
        assert post_result.output_data["injected_a"] == "val_a"
        assert post_result.output_data["injected_b"] == "val_b"


# ===========================================================================
# MiddlewarePipeline — Error Resilience
# ===========================================================================


class TestErrorResilience:

    def test_pre_process_error_skips_and_continues(self, base_context, caplog):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("before"),
            FailingPreMiddleware(),
            AnnotatingMiddleware("after"),
        ])
        with caplog.at_level(logging.ERROR):
            result = pipeline.execute_pre(base_context)

        # "before" and "after" should have run; FailingPreMiddleware skipped
        assert result.middleware_state.get("pre_before") is True
        assert result.middleware_state.get("pre_after") is True
        assert "FailingPreMiddleware" in caplog.text
        assert "pre_process" in caplog.text

    def test_post_process_error_skips_and_continues(
        self, base_context, base_response, caplog
    ):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("outer"),
            FailingPostMiddleware(),
            AnnotatingMiddleware("inner"),
        ])
        with caplog.at_level(logging.ERROR):
            result = pipeline.execute_post(base_context, base_response)

        # Reverse order: inner -> FailingPost (skipped) -> outer
        assert result.middleware_state.get("post_outer") is True
        assert result.middleware_state.get("post_inner") is True
        assert "FailingPostMiddleware" in caplog.text
        assert "post_process" in caplog.text

    def test_all_middleware_fail_returns_original_context(self, base_context, caplog):
        pipeline = MiddlewarePipeline([
            FailingPreMiddleware(),
            FailingPreMiddleware(),
        ])
        with caplog.at_level(logging.ERROR):
            result = pipeline.execute_pre(base_context)

        # Should get back essentially the original context
        assert result.request_id == base_context.request_id
        assert result.input_data == base_context.input_data

    def test_all_middleware_fail_returns_original_response(
        self, base_context, base_response, caplog
    ):
        pipeline = MiddlewarePipeline([
            FailingPostMiddleware(),
            FailingPostMiddleware(),
        ])
        with caplog.at_level(logging.ERROR):
            result = pipeline.execute_post(base_context, base_response)

        assert result.output_data == base_response.output_data


# ===========================================================================
# MiddlewarePipeline — Middleware State Accumulation
# ===========================================================================


class TestMiddlewareStateAccumulation:

    def test_state_accumulates_through_pre_pipeline(self, base_context):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("step1"),
            AnnotatingMiddleware("step2"),
            AnnotatingMiddleware("step3"),
        ])
        result = pipeline.execute_pre(base_context)

        # All steps should have left their mark
        assert result.middleware_state["pre_step1"] is True
        assert result.middleware_state["pre_step2"] is True
        assert result.middleware_state["pre_step3"] is True
        assert result.middleware_state["pre_order"] == ["step1", "step2", "step3"]

    def test_state_accumulates_through_post_pipeline(
        self, base_context, base_response
    ):
        pipeline = MiddlewarePipeline([
            AnnotatingMiddleware("step1"),
            AnnotatingMiddleware("step2"),
        ])
        result = pipeline.execute_post(base_context, base_response)

        assert result.middleware_state["post_step1"] is True
        assert result.middleware_state["post_step2"] is True
        # Reverse order
        assert result.middleware_state["post_order"] == ["step2", "step1"]

    def test_initial_state_preserved_through_pipeline(self):
        ctx = MiddlewareContext(
            request_id="r1",
            task_name="task",
            input_data={},
            middleware_state={"existing_key": "existing_value"},
        )
        pipeline = MiddlewarePipeline([AnnotatingMiddleware("mw")])
        result = pipeline.execute_pre(ctx)

        assert result.middleware_state["existing_key"] == "existing_value"
        assert result.middleware_state["pre_mw"] is True

    def test_original_context_not_mutated(self, base_context):
        pipeline = MiddlewarePipeline([AnnotatingMiddleware("mutator")])
        _ = pipeline.execute_pre(base_context)

        # The original context should be untouched (frozen model)
        assert base_context.middleware_state == {}

    def test_original_response_not_mutated(self, base_context, base_response):
        pipeline = MiddlewarePipeline([AnnotatingMiddleware("mutator")])
        _ = pipeline.execute_post(base_context, base_response)

        # The original response should be untouched (frozen model)
        assert base_response.middleware_state == {}


# ===========================================================================
# MiddlewarePipeline — from_config
# ===========================================================================


class TestFromConfig:

    def test_empty_config_returns_empty_pipeline(self):
        registry = MagicMock()
        pipeline = MiddlewarePipeline.from_config([], registry)
        assert len(pipeline.middlewares) == 0

    def test_none_config_returns_empty_pipeline(self):
        registry = MagicMock()
        pipeline = MiddlewarePipeline.from_config(None, registry)
        assert len(pipeline.middlewares) == 0

    def test_single_entry_config(self):
        mw = PassthroughMiddleware()
        registry = MagicMock()
        registry.create.return_value = mw

        config = [{"name": "passthrough", "config": {}}]
        pipeline = MiddlewarePipeline.from_config(config, registry)

        registry.create.assert_called_once_with("passthrough")
        assert len(pipeline.middlewares) == 1

    def test_multiple_entries_config(self):
        mw_a = AnnotatingMiddleware("a")
        mw_b = AnnotatingMiddleware("b")
        registry = MagicMock()
        registry.create.side_effect = [mw_a, mw_b]

        config = [
            {"name": "annotator_a", "config": {"tag": "a"}},
            {"name": "annotator_b", "config": {"tag": "b"}},
        ]
        pipeline = MiddlewarePipeline.from_config(config, registry)

        assert len(pipeline.middlewares) == 2
        assert registry.create.call_count == 2

    def test_config_with_kwargs_passed_to_registry(self):
        registry = MagicMock()
        registry.create.return_value = PassthroughMiddleware()

        config = [{"name": "pii_tokenizer", "config": {"pattern": r"\d{3}-\d{4}", "replace": "***"}}]
        MiddlewarePipeline.from_config(config, registry)

        registry.create.assert_called_once_with(
            "pii_tokenizer", pattern=r"\d{3}-\d{4}", replace="***"
        )

    def test_config_missing_config_key_uses_empty_dict(self):
        registry = MagicMock()
        registry.create.return_value = PassthroughMiddleware()

        config = [{"name": "simple"}]
        pipeline = MiddlewarePipeline.from_config(config, registry)

        registry.create.assert_called_once_with("simple")
        assert len(pipeline.middlewares) == 1

    def test_registry_error_skips_middleware(self, caplog):
        registry = MagicMock()
        registry.create.side_effect = [
            AnnotatingMiddleware("ok"),
            ValueError("unknown plugin"),
            AnnotatingMiddleware("also_ok"),
        ]

        config = [
            {"name": "good1", "config": {}},
            {"name": "bad", "config": {}},
            {"name": "good2", "config": {}},
        ]
        with caplog.at_level(logging.ERROR):
            pipeline = MiddlewarePipeline.from_config(config, registry)

        # The bad one should be skipped
        assert len(pipeline.middlewares) == 2
        assert "bad" in caplog.text

    def test_from_config_then_execute(self, base_context):
        """End-to-end: build from config, then execute pre-processing."""
        mw = AnnotatingMiddleware("configured")
        registry = MagicMock()
        registry.create.return_value = mw

        config = [{"name": "configured_mw", "config": {"tag": "configured"}}]
        pipeline = MiddlewarePipeline.from_config(config, registry)

        result = pipeline.execute_pre(base_context)
        assert result.middleware_state.get("pre_configured") is True


# ===========================================================================
# MiddlewarePipeline — middlewares property
# ===========================================================================


class TestMiddlewaresProperty:

    def test_middlewares_returns_copy(self):
        mw = PassthroughMiddleware()
        pipeline = MiddlewarePipeline([mw])
        mws = pipeline.middlewares
        mws.clear()  # mutate the returned list
        assert len(pipeline.middlewares) == 1  # internal list is unaffected

    def test_middlewares_preserves_order(self):
        mw_a = AnnotatingMiddleware("a")
        mw_b = AnnotatingMiddleware("b")
        mw_c = AnnotatingMiddleware("c")
        pipeline = MiddlewarePipeline([mw_a, mw_b, mw_c])
        mws = pipeline.middlewares
        assert mws[0] is mw_a
        assert mws[1] is mw_b
        assert mws[2] is mw_c
