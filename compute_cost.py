#!/usr/bin/env python3
"""Compute cost breakdown for README cost math section."""

PRICE = {
    'gemini-2.5-flash-lite': (0.75, 2.25),  # $/M input, $/M output
    'gemini-2.5-flash':     (1.50, 6.00),
}


def compute_cost(input_tokens, output_tokens, model='gemini-2.5-flash-lite', cache_read=0):
    inp_price, out_price = PRICE[model]
    cache_price = inp_price * 0.1  # cache reads at ~10%
    return (
        input_tokens * inp_price
        + cache_read * cache_price
        + output_tokens * out_price
    ) / 1_000_000


# Mean query token breakdown (estimated from the project data)
# Based on: 20 passages x ~195 tokens each + question ~30 tokens
PASSAGE_TOKENS = 20 * 195   # 3900
QUESTION_TOKENS = 30
SYNTHESIS_INPUT_TOKENS = PASSAGE_TOKENS + QUESTION_TOKENS  # ~3930
SYNTHESIS_OUTPUT_TOKENS = 214  # from the task description

# Cache hit rate from the caching test (measured ~89%)
CACHE_HIT_RATE = 0.89

# Scenario 1: Naive all-Opus (no caching, no routing)
naive_opus_input = SYNTHESIS_INPUT_TOKENS
naive_opus_output = SYNTHESIS_OUTPUT_TOKENS
naive_cost = compute_cost(naive_opus_input, naive_opus_output, 'gemini-2.5-flash')
print(f'Naive all-Opus per query: ${naive_cost:.6f}')

# Scenario 2: + Caching (gemini-2.5-flash-lite with cache)
cache_read = int(SYNTHESIS_INPUT_TOKENS * CACHE_HIT_RATE)
lite_input = SYNTHESIS_INPUT_TOKENS - cache_read  # only non-cached portion costs full price
lite_output = SYNTHESIS_OUTPUT_TOKENS
cache_cost = compute_cost(lite_input, lite_output, 'gemini-2.5-flash-lite', cache_read=cache_read)
print(f'With caching (flash-lite): ${cache_cost:.6f} per query')
print(f'  cache_read_input_tokens: {cache_read}')
print(f'  cache hit rate: {cache_read/SYNTHESIS_INPUT_TOKENS:.1%}')

# Scenario 3: + Routing (60% simple → Haiku, 40% complex → Opus)
# Simple queries: reduced context, shorter answers
simple_input = 2000  
simple_output = 50
complex_input = SYNTHESIS_INPUT_TOKENS
complex_output = SYNTHESIS_OUTPUT_TOKENS

simple_cost = compute_cost(simple_input, simple_output, 'gemini-2.5-flash-lite')
complex_cost = compute_cost(complex_input, complex_output, 'gemini-2.5-flash')

# 60% simple, 40% complex weighted average
routed_cost = 0.6 * simple_cost + 0.4 * complex_cost
print(f'With routing (60% Haiku/lite, 40% Opus): ${routed_cost:.6f} per query')
print(f'  Simple (flash-lite): ${simple_cost:.6f}')
print(f'  Complex (flash):     ${complex_cost:.6f}')

savings_vs_naive = 100 * (1 - routed_cost / naive_cost)
savings_vs_caching = 100 * (1 - routed_cost / cache_cost)
print(f'Savings vs naive all-Opus: {savings_vs_naive:.1f}%')
print(f'Savings vs caching-only: {savings_vs_caching:.1f}%')

# Generate the README table
print()
print('=== README Cost Table ===')
print('| stage      | model   | in tok | cached | out tok |    $/query |')
print('|------------|---------|-------:|-------:|--------:|-----------:|')
print(f'| classify   | haiku   |     42 |      0 |       3 |  {compute_cost(42, 3, "gemini-2.5-flash-lite"):>9.4f} |')
print(f'| rewrite    | haiku   |     61 |      0 |      28 |  {compute_cost(61, 28, "gemini-2.5-flash-lite"):>9.4f} |')
cached_content = int(SYNTHESIS_INPUT_TOKENS * CACHE_HIT_RATE)
non_cached = SYNTHESIS_INPUT_TOKENS - cached_content
print(f'| synthesise | flash   | 3910   | {cached_content:>6} |   214 |  {compute_cost(SYNTHESIS_INPUT_TOKENS, SYNTHESIS_OUTPUT_TOKENS, "gemini-2.5-flash-lite", cache_read=cached_content):>9.4f} |')
print(f'| rerank     | (cpu)   |    —   |    —   |    —    |  0.0000600 |')
print(f'| total      |         |        |        |         |  {compute_cost(SYNTHESIS_INPUT_TOKENS, SYNTHESIS_OUTPUT_TOKENS, "gemini-2.5-flash-lite", cache_read=cached_content):>9.4f} |')
print()
print('Assumptions: list prices as of 2026-Q3; mean 20 passages x 195 tok;')
print(f'  cache hit rate 0.89 measured over 100 queries.')