# Contributing

Thanks for helping improve XTLLM.

1. Open an issue before a large architectural change.
2. Keep model-specific policies model-specific when measurement supports them.
3. Do not add model weights, converted containers, proprietary SDK binaries,
   or benchmark outputs containing private prompts.
4. Build with warnings enabled and run a short exact-token regression.
5. For performance changes, report hardware, driver, command, tok/s, timed
   transitions, RAM/VRAM, SSD/H2D bytes per output, and before/after token IDs.
6. Preserve finite submissions, bounded waits, authoritative routing, and the
   no-TDR-change rule.

Pull requests should be small enough to review and must explain any quality or
memory trade-off. Formatting-only changes should not be mixed with kernel work.

By contributing, you agree that your contribution is licensed under Apache-2.0.
