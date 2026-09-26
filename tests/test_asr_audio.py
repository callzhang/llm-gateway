import numpy as np
import pytest

from asr_adapter import audio
from asr_adapter.audio import SAMPLE_RATE, Chunk, is_degenerate, join_text, make_chunks, plan_cuts, frame_rms, token_budget


def tone(seconds, amp=0.1):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def test_short_recording_is_a_single_chunk_trimmed_to_its_voiced_span():
    x = np.concatenate([silence(3), tone(5), silence(4)])
    (chunk,) = make_chunks(x)
    assert chunk.start == pytest.approx(2.7, abs=0.11)   # 0.3 s of context before the voice
    assert chunk.end == pytest.approx(8.3, abs=0.11)
    assert len(chunk.samples) == round((chunk.end - chunk.start) * SAMPLE_RATE)


def test_pure_silence_and_near_silence_yield_no_chunks():
    assert make_chunks(silence(40)) == []
    assert make_chunks(np.full(SAMPLE_RATE * 10, 0.001, dtype=np.float32)) == []
    assert make_chunks(np.zeros(0, dtype=np.float32)) == []


def test_a_mostly_silent_clip_keeps_only_the_speech_not_the_silence():
    # 42 s of digital silence then 15 s of speech (the clip that made a model loop)
    x = np.concatenate([silence(42), tone(15)])
    chunks = make_chunks(x)
    assert chunks and all(c.end - c.start < 32 for c in chunks)
    assert chunks[0].start >= 41.0


def test_long_recording_is_cut_at_quiet_points_near_the_target_length():
    piece = np.concatenate([tone(4.6), silence(0.4)])
    x = np.tile(piece, 30)                                # 150 s of speech with a pause every 5 s
    chunks = make_chunks(x, target_s=30, slack_s=4)
    assert 4 <= len(chunks) <= 6
    for chunk in chunks[:-1]:
        assert 26 <= chunk.end - chunk.start <= 34.5
    # cuts fall in the pauses: no chunk boundary lands inside a tone burst
    rms = frame_rms(x)
    for cut in plan_cuts(rms)[1:]:
        assert rms[cut] < audio.VOICE_RMS


def test_chunks_are_ordered_and_do_not_overlap():
    x = np.tile(np.concatenate([tone(4.6), silence(0.4)]), 40)
    chunks = make_chunks(x)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start + 1e-6
    assert chunks[-1].end == pytest.approx(len(x) / SAMPLE_RATE, abs=0.2)


def test_is_degenerate_flags_loops_but_not_speech():
    assert is_degenerate("我就是想说，" * 40)
    assert is_degenerate("我们是祖国的花朵，阳光下尽情唱着歌。" * 12)
    assert not is_degenerate("对对对对对")
    assert not is_degenerate("好，然后我昨天主要是配合着 memory 迁移的那个测试，然后给包括给服务这边的 bug 和 memory bug 都已经提给他们，然后再改。")
    assert not is_degenerate("")
    normal = "今天的计划呢，首先看一下俊杰能不能接好，如果有什么困难的话及时上升，然后还有就是要跟小明哥去对一下下一步网关的一个设计。" * 2
    assert not is_degenerate(normal)


def test_join_text_only_spaces_between_ascii_words():
    assert join_text(["你好", "世界"]) == "你好世界"
    assert join_text(["hello", "world"]) == "hello world"
    assert join_text(["测试 memory", "bug 已经修了"]) == "测试 memory bug 已经修了"
    assert join_text([" ", "", "你好"]) == "你好"


def test_token_budget_scales_with_duration_and_never_hits_zero():
    assert token_budget(0) == 64
    assert token_budget(30) == 304
    assert token_budget(60) > token_budget(30)
