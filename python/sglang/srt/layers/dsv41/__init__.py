"""Pure-torch reference implementation of the DeepSeek V4.1 layers.

Stateful in the reference style (start_pos, per-module caches) and built from
plain torch ops, so it is the numerical oracle for the production path in
layers/attention/dsv4, layers/moe, layers/quantization and models/deepseek_v41,
not a serving path itself.
"""
