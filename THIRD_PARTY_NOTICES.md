# Third-party sources

The selected reference files under references/learn-claude-code are from [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code), commit ce8f9f186058939da54c9d6fead78dfb5d0fd6c3.

Copyright (c) 2024 shareAI Lab. Distributed under the MIT License; full text is retained in [LICENSE](references/learn-claude-code/LICENSE).

tests/test_upstream_compact_portable.py derives from that commit's tests/test_s08_context_compact.py. Changes are limited to the reference source location and platform-neutral path comparison. All other test assertions remain as upstream. This derived test is also covered by the retained MIT notice.

src/paper_agent/runtime.py adapts the s02 message/tool/result loop at the same commit; it adds injectable clients, schema validation, bounded calls, hooks and usage tracking. Keep the upstream MIT notice with distributions. src/paper_agent/memory.py uses s08's archive/preview concept; the session state and citation validation are new domain code. The seven-stage pipeline is new, not an upstream capability.

Direct dependencies: FlagEmbedding (MIT), FAISS (MIT), rank_bm25 (Apache-2.0), pypdf (BSD-3-Clause), jsonschema (MIT), NumPy (BSD-3-Clause), plus their transitive packages. Their packages retain their own license files. RRF's rank formula follows the Pyserini reference; no Pyserini runtime is imported.

Dataset: QASPER v0.3, Dasigi et al., NAACL 2021, https://aclanthology.org/2021.naacl-main.365/ . Official dataset metadata declares CC BY 4.0. Download source and archive hash are recorded in data/qasper/manifest.json. No claim is made of local human review. Model licenses and revisions are recorded alongside the original BAAI model cards in models/ after successful download.

API configuration uses python-dotenv (BSD-3-Clause). The new providers.py implements the Chat Completions HTTP contract directly, with local JSON Schema validation; it does not copy a third-party gateway implementation. Endpoint/JSON-mode references checked on 2026-10-22: [DeepSeek API](https://api-docs.deepseek.com/), [DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode/), [OpenCode harness endpoints](https://opencode.ai/v2/docs/console/harnesses), [Go usage scope](https://opencode.ai/v2/docs/console/go).

data/dialogues-draft contains template-derived prompts referencing the QASPER subset. It retains source question/paper IDs and explicitly marks all follow-ups and provisional labels unreviewed. It is not the original screenshot's evaluation set and is not independently annotated gold.
