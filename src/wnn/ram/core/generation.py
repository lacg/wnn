"""
Generation Strategies for Seq2Seq Models

Provides different decoding strategies for autoregressive generation:
- Greedy: Always pick the highest probability token
- Beam Search: Maintain multiple candidates, select best sequences
- Sampling: Sample from probability distribution
- Top-k/Top-p: Constrained sampling

Streaming versions (generators) are also available:
- stream_greedy_decode: Yields tokens as they are generated
- stream_sample_decode: Yields sampled tokens
- stream_top_k_decode: Yields top-k sampled tokens

Usage:
    # Greedy generation
    output = greedy_decode(model, encoder_output, max_len=10)

    # Beam search
    output = beam_search(model, encoder_output, beam_width=3, max_len=10)

    # Streaming greedy (yields tokens one by one)
    for token in stream_greedy_decode(model, encoder_output, max_len=10):
        print(f"Generated: {token}")
"""

from typing import Callable, Iterator, Generator
from dataclasses import dataclass
from torch import Tensor, zeros, uint8, stack, cat, float32, argsort


@dataclass
class BeamCandidate:
    """A single beam search candidate."""
    tokens: list[Tensor]  # Generated tokens so far
    score: float          # Log probability (higher = better)
    finished: bool        # Whether sequence is complete


@dataclass
class GenerationResult:
    """Result of sequence generation."""
    tokens: list[Tensor]    # Generated token sequence
    score: float            # Log probability score
    candidates: list[BeamCandidate] | None = None  # All beam candidates (optional)


def greedy_decode(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    max_len: int,
    eos_value: int | None = None,
    get_score: Callable[[Tensor], float] | None = None,
) -> GenerationResult:
    """
    Greedy decoding: always pick the most likely next token.

    Args:
        model: Decoder model that takes (tokens, encoder_output) -> next_tokens
               The model should be autoregressive (uses past tokens as input)
        encoder_output: Encoder hidden states (None for decoder-only)
        start_token: Start-of-sequence token
        max_len: Maximum generation length
        eos_value: End-of-sequence token value (stops generation)
        get_score: Function to score tokens (for logging only)

    Returns:
        GenerationResult with generated sequence
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int

    generated = [start_token.clone()]
    total_score = 0.0

    for step in range(max_len):
        # Get next token prediction
        output = model(generated, encoder_output)
        next_token = output[-1].clone()  # Last position output

        # Score (if provided)
        if get_score:
            total_score += get_score(next_token)

        generated.append(next_token)

        # Check EOS
        if eos_value is not None:
            token_val = bits_to_int(next_token)
            if token_val == eos_value:
                break

    return GenerationResult(
        tokens=generated[1:],  # Exclude start token
        score=total_score,
    )


def beam_search(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    beam_width: int,
    max_len: int,
    vocab_size: int | None = None,
    eos_value: int | None = None,
    length_penalty: float = 0.0,
    score_fn: Callable[[Tensor, Tensor], float] | None = None,
) -> GenerationResult:
    """
    Beam search decoding: maintain multiple candidate sequences.

    Args:
        model: Decoder model (autoregressive)
        encoder_output: Encoder hidden states (None for decoder-only)
        start_token: Start-of-sequence token
        beam_width: Number of beams to maintain
        max_len: Maximum generation length
        vocab_size: Vocabulary size (2^token_bits)
        eos_value: End-of-sequence token value
        length_penalty: Penalize shorter sequences (0.0 = no penalty)
        score_fn: Custom scoring function(output_token, all_outputs) -> score
                  If None, uses Hamming similarity to output

    Returns:
        GenerationResult with best sequence and all candidates
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int, int_to_bits

    token_bits = len(start_token)
    vocab_size = vocab_size or (2 ** token_bits)

    # Initialize beams with start token
    beams = [
        BeamCandidate(
            tokens=[start_token.clone()],
            score=0.0,
            finished=False,
        )
    ]

    for step in range(max_len):
        all_candidates = []

        for beam in beams:
            if beam.finished:
                all_candidates.append(beam)
                continue

            # Get model prediction
            output = model(beam.tokens, encoder_output)
            last_output = output[-1]

            # Score all possible next tokens
            for token_val in range(min(vocab_size, 2 ** token_bits)):
                candidate_token = int_to_bits(token_val, token_bits)

                # Score: Hamming similarity (more matching bits = higher score)
                if score_fn:
                    token_score = score_fn(candidate_token, last_output)
                else:
                    # Default: Hamming similarity
                    xor = candidate_token ^ last_output
                    matching = token_bits - int(xor.sum().item())
                    token_score = matching / token_bits  # Normalize to [0, 1]

                new_tokens = beam.tokens + [candidate_token]
                new_score = beam.score + token_score

                # Length penalty
                if length_penalty > 0:
                    new_score = new_score / (len(new_tokens) ** length_penalty)

                finished = (eos_value is not None and token_val == eos_value)

                all_candidates.append(BeamCandidate(
                    tokens=new_tokens,
                    score=new_score,
                    finished=finished,
                ))

        # Select top-k candidates
        all_candidates.sort(key=lambda c: -c.score)
        beams = all_candidates[:beam_width]

        # Early stopping if all beams finished
        if all(b.finished for b in beams):
            break

    # Return best beam
    best = max(beams, key=lambda b: b.score)

    return GenerationResult(
        tokens=best.tokens[1:],  # Exclude start token
        score=best.score,
        candidates=beams,
    )


def sample_decode(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    max_len: int,
    temperature: float = 1.0,
    eos_value: int | None = None,
) -> GenerationResult:
    """
    Sampling-based decoding with temperature.

    For RAM models, we sample based on output bit confidence
    (treating output as soft probabilities).

    Args:
        model: Decoder model
        encoder_output: Encoder hidden states
        start_token: Start token
        max_len: Maximum length
        temperature: Sampling temperature (higher = more random)
        eos_value: End-of-sequence value

    Returns:
        GenerationResult with sampled sequence
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int
    import random

    generated = [start_token.clone()]
    total_score = 0.0

    for step in range(max_len):
        output = model(generated, encoder_output)
        last_output = output[-1]

        # For RAM: output is binary, sample based on bit values
        # If temperature > 1, add noise
        if temperature > 1.0:
            noise_prob = 1.0 - (1.0 / temperature)
            noisy_output = last_output.clone()
            for i in range(len(noisy_output)):
                if random.random() < noise_prob:
                    noisy_output[i] = 1 - noisy_output[i]  # Flip bit
            next_token = noisy_output
        else:
            next_token = last_output.clone()

        generated.append(next_token)

        # Check EOS
        if eos_value is not None:
            if bits_to_int(next_token) == eos_value:
                break

    return GenerationResult(
        tokens=generated[1:],
        score=total_score,
    )


def top_k_decode(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    max_len: int,
    k: int = 5,
    eos_value: int | None = None,
) -> GenerationResult:
    """
    Top-k decoding: sample from top k most likely tokens.

    For RAM models, we find the k tokens closest to the output
    (by Hamming distance) and sample uniformly from them.

    Args:
        model: Decoder model
        encoder_output: Encoder hidden states
        start_token: Start token
        max_len: Maximum length
        k: Number of top candidates to sample from
        eos_value: End-of-sequence value

    Returns:
        GenerationResult
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int, int_to_bits
    import random

    token_bits = len(start_token)
    vocab_size = 2 ** token_bits

    generated = [start_token.clone()]

    for step in range(max_len):
        output = model(generated, encoder_output)
        last_output = output[-1]

        # Score all tokens by Hamming similarity
        scores = []
        for token_val in range(vocab_size):
            candidate = int_to_bits(token_val, token_bits)
            xor = candidate ^ last_output
            similarity = token_bits - int(xor.sum().item())
            scores.append((similarity, token_val))

        # Sort by similarity (descending) and take top-k
        scores.sort(key=lambda x: -x[0])
        top_k = scores[:k]

        # Sample uniformly from top-k
        _, chosen_val = random.choice(top_k)
        next_token = int_to_bits(chosen_val, token_bits)

        generated.append(next_token)

        # Check EOS
        if eos_value is not None and chosen_val == eos_value:
            break

    return GenerationResult(
        tokens=generated[1:],
        score=0.0,
    )


# =============================================================================
# Streaming Generation (Generators)
# =============================================================================

@dataclass
class StreamToken:
    """A single token from streaming generation."""
    token: Tensor       # The generated token
    step: int           # Generation step (0-indexed)
    is_eos: bool        # Whether this is an end-of-sequence token
    score: float = 0.0  # Token score (if available)


def stream_greedy_decode(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    max_len: int,
    eos_value: int | None = None,
    get_score: Callable[[Tensor], float] | None = None,
) -> Generator[StreamToken, None, None]:
    """
    Streaming greedy decoding: yields tokens as they are generated.

    This is a generator version of greedy_decode that allows processing
    tokens as they are generated rather than waiting for the full sequence.

    Args:
        model: Decoder model that takes (tokens, encoder_output) -> next_tokens
        encoder_output: Encoder hidden states (None for decoder-only)
        start_token: Start-of-sequence token
        max_len: Maximum generation length
        eos_value: End-of-sequence token value (stops generation)
        get_score: Function to score tokens

    Yields:
        StreamToken for each generated token

    Example:
        for token_info in stream_greedy_decode(model, None, start, max_len=10):
            print(f"Step {token_info.step}: {token_info.token}")
            if some_condition:
                break  # Can stop early
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int

    generated = [start_token.clone()]

    for step in range(max_len):
        # Get next token prediction
        output = model(generated, encoder_output)
        next_token = output[-1].clone()

        # Score (if provided)
        score = get_score(next_token) if get_score else 0.0

        # Check EOS
        is_eos = False
        if eos_value is not None:
            token_val = bits_to_int(next_token)
            is_eos = (token_val == eos_value)

        # Yield the token
        yield StreamToken(
            token=next_token,
            step=step,
            is_eos=is_eos,
            score=score,
        )

        # Stop if EOS
        if is_eos:
            return

        generated.append(next_token)


def stream_sample_decode(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    max_len: int,
    temperature: float = 1.0,
    eos_value: int | None = None,
) -> Generator[StreamToken, None, None]:
    """
    Streaming sampling-based decoding with temperature.

    Args:
        model: Decoder model
        encoder_output: Encoder hidden states
        start_token: Start token
        max_len: Maximum length
        temperature: Sampling temperature (higher = more random)
        eos_value: End-of-sequence value

    Yields:
        StreamToken for each sampled token
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int
    import random

    generated = [start_token.clone()]

    for step in range(max_len):
        output = model(generated, encoder_output)
        last_output = output[-1]

        # For RAM: output is binary, sample based on bit values
        if temperature > 1.0:
            noise_prob = 1.0 - (1.0 / temperature)
            noisy_output = last_output.clone()
            for i in range(len(noisy_output)):
                if random.random() < noise_prob:
                    noisy_output[i] = 1 - noisy_output[i]
            next_token = noisy_output
        else:
            next_token = last_output.clone()

        # Check EOS
        is_eos = False
        if eos_value is not None:
            is_eos = (bits_to_int(next_token) == eos_value)

        yield StreamToken(
            token=next_token,
            step=step,
            is_eos=is_eos,
        )

        if is_eos:
            return

        generated.append(next_token)


def stream_top_k_decode(
    model: Callable[[list[Tensor], list[Tensor] | None], list[Tensor]],
    encoder_output: list[Tensor] | None,
    start_token: Tensor,
    max_len: int,
    k: int = 5,
    eos_value: int | None = None,
) -> Generator[StreamToken, None, None]:
    """
    Streaming top-k decoding: yields sampled tokens from top k candidates.

    Args:
        model: Decoder model
        encoder_output: Encoder hidden states
        start_token: Start token
        max_len: Maximum length
        k: Number of top candidates to sample from
        eos_value: End-of-sequence value

    Yields:
        StreamToken for each generated token
    """
    from wnn.ram.core.transformers.computed_arithmetic import bits_to_int, int_to_bits
    import random

    token_bits = len(start_token)
    vocab_size = 2 ** token_bits

    generated = [start_token.clone()]

    for step in range(max_len):
        output = model(generated, encoder_output)
        last_output = output[-1]

        # Score all tokens by Hamming similarity
        scores = []
        for token_val in range(vocab_size):
            candidate = int_to_bits(token_val, token_bits)
            xor = candidate ^ last_output
            similarity = token_bits - int(xor.sum().item())
            scores.append((similarity, token_val))

        # Sort by similarity (descending) and take top-k
        scores.sort(key=lambda x: -x[0])
        top_k = scores[:k]

        # Sample uniformly from top-k
        best_sim, chosen_val = random.choice(top_k)
        next_token = int_to_bits(chosen_val, token_bits)

        # Check EOS
        is_eos = (eos_value is not None and chosen_val == eos_value)

        yield StreamToken(
            token=next_token,
            step=step,
            is_eos=is_eos,
            score=best_sim / token_bits,
        )

        if is_eos:
            return

        generated.append(next_token)


def collect_stream(
    stream: Generator[StreamToken, None, None],
) -> GenerationResult:
    """
    Collect all tokens from a streaming generator into a GenerationResult.

    Useful when you want to use streaming internally but return a result.

    Args:
        stream: A streaming generation generator

    Returns:
        GenerationResult with all collected tokens
    """
    tokens = []
    total_score = 0.0

    for token_info in stream:
        tokens.append(token_info.token)
        total_score += token_info.score

    return GenerationResult(
        tokens=tokens,
        score=total_score,
    )
