---
name: Signal/condition type layers
description: Where a new strategy signal/condition type must be registered so it works end-to-end (executor + AI + web + mobile)
---

# Adding a new signal / condition type end-to-end

A new strategy signal type only works if its config-shape (the `condition`/param
keys) is registered identically across FOUR independent layers. The executor is
the source of truth for key names — every other layer must emit exactly what it
reads.

**Why:** these layers have no shared schema; a key mismatch (e.g. wizard emits
`std` but executor reads `num_std`) silently produces a config the executor
can't evaluate, with no compile error.

**How to apply** — register in all of:
1. **Execution engine** `app/services/strategy_ta.py` — add `eval_<type>()` +
   a dispatch branch in `_eval_one`. This defines the canonical config keys.
2. **AI chat builder** — `app/services/strategy_builder.py` `CONDITION_SCHEMA`
   (the registry the AI's `###STRATEGY###` NL is compiled against) AND the
   prompt text + `/api/generate-indicator` CONDITION REFERENCE in
   `strategy_portal_server.py`. (`ai_strategy_generator.py` is marketplace-only,
   out of scope.)
3. **Web wizard** `app/templates/strategy_portal.html` — `SIGNAL_META`,
   `SIGNAL_FREQ`, `getDefaultCfg`, `renderSignalCfg`, and the NL serialization in
   `buildWizardDescription` (primary + confirmations).
4. **Mobile wizard** — `mobile/lib/strategyPresets.ts` (`SignalType` union +
   `SIGNAL_META` + `getDefaultCfg`), `mobile/lib/wizardConfig.ts`
   (`describeSignal`), `mobile/components/wizard/ConditionEditor.tsx`
   (`renderKnobs` case). The mobile `getDefaultCfg` switch is exhaustive over
   `SignalType` with no default, so adding a union member forces adding a case —
   tsc catches a missed one. `describeSignal` and `renderKnobs` DO have a
   default, so they fail silently if a case is missed — add them deliberately.

Top-level executor types pass straight through `packCondition` in
`mobile/lib/conditionPack.ts` via `{ type, timeframe, ...cfg }` — do NOT add them
to `INDICATOR_NAME_MAP` (that's only for sub-indicators nested under a generic
`indicator` type).

## Two gotchas when the new type is an ICT/`fx_*` family member
- **The web wizard does NOT expose the `fx_*` ICT family at all** (no
  `fx_killzone`/`fx_displacement`/`fx_cisd`/etc. in `strategy_portal.html`). Those
  types are reachable ONLY via the AI chat builder and the mobile wizard. So for a
  new `fx_*` type, layer 3 (web wizard) is intentionally SKIPPED — forcing it in
  would be inconsistent with every sibling type.
- **`strategy_builder.py` has TWO independent places to register an `fx_*` type**,
  and the AI is unreliable if you update only one: (1) the `CONDITION_SCHEMA`
  reference block (the `{"type":...}` JSON schema near the other ICT entries), AND
  (2) a separate prose "ICT / Day-trading signal mappings (forex)" block
  (`• "phrase" / "synonyms" → fx_type`). Add the synonym line to BOTH, plus the
  `SIGNAL RECOGNITION` + `CONDITION REFERENCE` blocks in `strategy_portal_server.py`.

Verify (dev executor is prod-only / disabled): `python -m py_compile` the 3
backend files, `cd mobile && npx tsc --noEmit`, restart the Strategy Portal
workflow and confirm HTTP 200 — not live signal firing.
