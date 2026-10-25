Local changes to github.com/andyjliu/llm-agent-collusion used for the double-auction work.
Apply from a clean clone of that repo: `git apply llm-agent-collusion.patch && cp mock_wrapper.py src/resources/`.
Adds: OpenRouter routing for any "/"-style model id, a free mock client, ZIP buyers ("zip") with working adaptation,
partial-result saving on abort, opt-in LLM judge, --colluding_sellers / --private_channel / --shadow_twins,
JSON mode for openai/* via OpenRouter, raw-response logging on JSON failures. See results/double_auction/ANALYSIS_LOG.md §8.
