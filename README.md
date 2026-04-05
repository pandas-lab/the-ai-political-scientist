# The Agentic Political Scientist
_Vibe Science_

This repository contains the prompts, supplemental materials, and output from experiments in the use of local Large Language Models (LLMs) to autonomously generate research products in political science and adjacent fields.

## Authors
-    [Benjamin J. Radford](https://www.benradford.com)
-    Your name here if you participate

## Contents
1. [Submitting a Project](#submitting-a-project)
2. [Rules](#rules)
3. [Recommended Models](#recommended-models)
4. [How Does This Work?](#how-does-this-work?)
5. [Notes](#notes)

## Submitting a Project

If you are a current member the University of Pittsburgh's Department of Political Science or School of Public and International Affairs, send to me ([Benjamin J. Radford](https://www.benradford.com)) the following items:

1. **Prompt**: Your prompt as a text document.
    - It can be any length that you want, from a few sentences to several pages.
    - It can be as detailed as you like.
    - You may prompt the AI to use tools (though it will on its own, too).
2. **Supplemental Materials (optional)**: Any additional materials you want the AI to have access to.
    - Datasets.
    - Code you've written.
    - Papers you want it to read for inspiration.
3. **Model Choice (optional)**: Which local model, from [Ollama](https://ollama.com/search), do you want to use?
    - If you don't pick one, I'll just use my favorite.
    - It must be one that has the `tools` and `thinking` tags.
    - It cannot be one that _only_ has `cloud` versions (it must have a downloadable version).
    - If it's too big or I can't get it to work, I'll fall back on my favorite model.
  
## Rules

- You may not use any generated papers as coursework.
- All prompts, supporting materials, and generated papers will be published in this repository unless you specifically ask me not to.
- I will provide no guidance to the LLM beyond providing your prompt and approving tool use and file edits.

## Recommended Models

- [qwen3-coder-next](https://ollama.com/library/qwen3-coder-next)
- [qwen3.5-27B](https://ollama.com/library/qwen3.5)
- [glm-4.7-flash](https://ollama.com/library/glm-4.7-flash)
- [gpt-oss](https://ollama.com/library/gpt-oss)
- [nemotron-cascade-2](https://ollama.com/library/nemotron-cascade-2)
- I'm working on the new hotness, [Gemma 4](https://ollama.com/library/gemma4), but it's tool use capabilities are currently broken.

## How Does This Work?

I have both Claude Code (harness) and Ollama (LLM server) running on my workstation. Ollama let's us pick a local LLM to run and Claude Code provides that LLM with access to my computer and a variety of tools (web search, R, Python, LaTeX, etc...). It is, effectively, Claude Code but with open-source or open-weight models rather than Anthropic's expensive and proprietary models (Haiku, Sonnet, or Opus). These are essentially free for us but are typically not as capable as Anthropic's offerings. 

## Notes

- Check out the Sonnet 4.6 paper. It's legit. You get what you pay for.
