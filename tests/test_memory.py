"""Tests for the Memory message store."""

from __future__ import annotations

import pytest

from pyagent import Memory


class TestAppendAndCombine:
    def test_add_appends_message(self) -> None:
        memory = Memory()
        result = memory.add({"role": "user", "content": "hello"})
        assert result is memory
        assert memory.messages == [{"role": "user", "content": "hello"}]

    def test_append_alias(self) -> None:
        memory = Memory()
        memory.append({"role": "user", "content": "hi"})
        assert len(memory) == 1

    def test_plus_operator_with_dict_returns_new_memory(self) -> None:
        memory = Memory(system="sys")
        memory.add({"role": "user", "content": "hi"})
        combined = memory + {"role": "assistant", "content": "hello"}
        assert len(memory) == 2
        assert len(combined) == 3
        assert combined.messages[-1] == {"role": "assistant", "content": "hello"}

    def test_plus_operator_merges_memory(self) -> None:
        first = Memory(system="sys")
        first.add({"role": "user", "content": "a"})
        second = Memory()
        second.add({"role": "tool", "tool_call_id": "add", "content": "5"})
        merged = first + second
        assert [message["role"] for message in merged.messages] == [
            "system",
            "user",
            "tool",
        ]

    def test_iadd_mutates_in_place(self) -> None:
        memory = Memory()
        memory += {"role": "user", "content": "yo"}
        assert len(memory) == 1
        memory2 = Memory(system="s")
        memory += memory2
        assert [message["role"] for message in memory.messages] == ["user", "system"]

    def test_plus_rejects_other_types(self) -> None:
        memory = Memory()
        with pytest.raises(TypeError):
            memory + 42  # type: ignore[operator]


class TestSystem:
    def test_init_with_system_message(self) -> None:
        memory = Memory(system="be terse")
        assert memory.messages == [{"role": "system", "content": "be terse"}]

    def test_set_system_replaces_existing(self) -> None:
        memory = Memory(system="first")
        memory.add({"role": "user", "content": "u"})
        memory.set_system("second")
        assert memory.messages[0] == {"role": "system", "content": "second"}
        assert len(memory.messages) == 2

    def test_clear_keeps_system(self) -> None:
        memory = Memory(system="sys")
        memory.add({"role": "user", "content": "u"})
        memory.clear()
        assert memory.messages == [{"role": "system", "content": "sys"}]


class TestTrim:
    def test_keeps_system_and_last_n(self) -> None:
        memory = Memory(system="sys")
        for index in range(10):
            memory.add({"role": "user", "content": str(index)})
        memory.trim(3)
        contents = [message["content"] for message in memory.messages]
        assert contents == ["sys", "7", "8", "9"]

    def test_noop_when_within_limit(self) -> None:
        memory = Memory(system="sys")
        memory.add({"role": "user", "content": "u"})
        memory.trim(4)
        assert len(memory) == 2

    def test_is_chainable(self) -> None:
        memory = Memory()
        memory.add({"role": "user", "content": "u"})
        assert memory.trim(1) is memory

    def test_rejects_non_positive_limit(self) -> None:
        memory = Memory()
        with pytest.raises(ValueError):
            memory.trim(0)


class TestSerialization:
    def test_json_roundtrip(self) -> None:
        memory = Memory(system="sys")
        memory.add({"role": "user", "content": "hello"})
        memory.add(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1}'},
                    }
                ],
            }
        )
        memory.add({"role": "tool", "tool_call_id": "call_1", "content": "5"})
        rebuilt = Memory.from_json(memory.to_json())
        assert isinstance(memory.to_json(), str)
        assert rebuilt.messages == memory.messages

    def test_from_json_handles_full_document(self) -> None:
        data = '[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]'
        memory = Memory.from_json(data)
        assert [message["role"] for message in memory.messages] == ["system", "user"]

    def test_from_json_invalid_string(self) -> None:
        with pytest.raises(ValueError):
            Memory.from_json("{not valid json")

    def test_from_json_rejects_non_list(self) -> None:
        with pytest.raises(ValueError):
            Memory.from_json('{"role": "user"}')  # type: ignore[arg-type]


class TestEstimateTokens:
    def test_empty_memory_is_zero(self) -> None:
        assert Memory().estimate_tokens() == 0

    def test_grows_with_more_content(self) -> None:
        short = Memory(system="hi")
        long = Memory(system="the quick brown fox jumps over the lazy dog")
        assert long.estimate_tokens() > short.estimate_tokens()

    def test_more_messages_cost_more(self) -> None:
        one = Memory()
        one.add({"role": "user", "content": "a word"})
        two = Memory()
        two.add({"role": "user", "content": "a word"})
        two.add({"role": "assistant", "content": "a reply"})
        assert two.estimate_tokens() > one.estimate_tokens()

    def test_urls_are_conservative(self) -> None:
        plain = Memory()
        plain.add({"role": "user", "content": "a b c"})
        urls = Memory()
        urls.add(
            {
                "role": "user",
                "content": "open https://example.com/top/mid/bottom/file.dat here",
            }
        )
        assert urls.estimate_tokens() > plain.estimate_tokens()

    def test_file_like_tokens_are_conservative(self) -> None:
        plain = Memory()
        plain.add({"role": "user", "content": "a b c"})
        file_content = Memory()
        file_content.add({"role": "user", "content": "read /etc/hosts and notes.txt please"})
        assert file_content.estimate_tokens() > plain.estimate_tokens()


class TestValidationAndIteration:
    def test_missing_role_rejected(self) -> None:
        memory = Memory()
        with pytest.raises(ValueError):
            memory.add({"content": "no role"})

    def test_unknown_role_rejected(self) -> None:
        memory = Memory()
        with pytest.raises(ValueError):
            memory.add({"role": "wizard", "content": "abracadabra"})

    def test_non_dict_rejected(self) -> None:
        memory = Memory()
        with pytest.raises(ValueError):
            memory.add("not a dict")  # type: ignore[arg-type]

    def test_iteration_and_indexing(self) -> None:
        memory = Memory(system="sys")
        memory.add({"role": "user", "content": "u"})
        assert [message["role"] for message in memory] == ["system", "user"]
        assert memory[1]["content"] == "u"
        assert bool(memory)
        assert not Memory()

    def test_messages_returns_copies(self) -> None:
        memory = Memory()
        memory.add({"role": "user", "content": "u"})
        snapshot = memory.messages
        snapshot[0]["content"] = "mutated"
        assert memory.messages[0]["content"] == "u"

    def test_repr_mentions_count(self) -> None:
        memory = Memory()
        memory.add({"role": "user", "content": "u"})
        assert "Memory(1 messages)" in repr(memory)