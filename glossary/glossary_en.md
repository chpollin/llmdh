# Glossary of Generative AI and Large Language Models

## Large Language Model (LLM)
id: llm
tags: fundamentals, architecture
level: basic

A Large Language Model is a type of artificial neural network trained on vast amounts of text data to understand and generate human-like language. LLMs like GPT, Claude, or Gemini use transformer architecture and are characterized by billions of parameters that encode statistical patterns about language. These models can perform a wide range of language tasks through prompting without task-specific training.

* Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762
* Brown et al. (2020). Language Models are Few-Shot Learners. https://arxiv.org/abs/2005.14165

## Context Window
id: context-window
tags: fundamentals, architecture
level: basic

The context window defines the maximum amount of text (measured in tokens) that an LLM can process in a single interaction. It includes both the input prompt and the generated response. Modern LLMs have context windows ranging from 8K tokens (approximately 6,000 words) to over 1 million tokens. A larger context window allows the model to maintain coherence over longer conversations and process larger documents.

* Anthropic (2024). Introducing Claude 3.5 Sonnet with 200K context window. https://www.anthropic.com/claude

## Prompt Engineering
id: prompt-engineering
tags: fundamentals, prompting
level: basic

Prompt engineering is the practice of designing and refining text inputs (prompts) to effectively communicate with and guide LLMs toward desired outputs. It involves understanding how models interpret instructions, providing clear context, using examples (few-shot learning), and iteratively improving prompts based on model responses. Effective prompt engineering can significantly improve output quality without modifying the model itself.

* Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903
* OpenAI (2023). Prompt Engineering Guide. https://platform.openai.com/docs/guides/prompt-engineering

## RAG (Retrieval-Augmented Generation)
id: rag
tags: ai-engineering, techniques
level: intermediate

RAG is a technique that enhances LLM outputs by combining them with external knowledge retrieval. When a query is received, relevant documents or passages are first retrieved from a knowledge base (often using vector similarity search), then provided to the LLM as context. This approach reduces hallucinations, allows models to access up-to-date information, and grounds responses in specific source materials. RAG is particularly valuable for domain-specific applications.

* Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401

## Token
id: token
tags: fundamentals, architecture
level: basic

A token is the basic unit of text that LLMs process. Tokenization breaks down text into smaller pieces—typically words, subwords, or characters. For example, "understanding" might be split into "under" and "standing" as two tokens. The number of tokens in a text affects processing cost and context window usage. On average, one token represents approximately 0.75 words in English.

* OpenAI (2023). What are tokens and how to count them? https://help.openai.com/en/articles/4936856

## Fine-tuning
id: fine-tuning
tags: training, techniques
level: intermediate

Fine-tuning is the process of taking a pre-trained LLM and further training it on a smaller, task-specific dataset to adapt it for particular use cases or domains. This process adjusts the model's parameters to optimize performance on the target task while retaining general language capabilities. Fine-tuning requires less data and computational resources than training from scratch but more than prompt engineering alone.

* Howard & Ruder (2018). Universal Language Model Fine-tuning for Text Classification. https://arxiv.org/abs/1801.06146

## Hallucination
id: hallucination
tags: fundamentals, safety, limitations
level: basic

Hallucination refers to when an LLM generates information that appears plausible but is factually incorrect or fabricated. This occurs because LLMs are trained to predict statistically likely text patterns rather than to verify factual accuracy. Hallucinations are a fundamental limitation of current LLM architectures and can be partially mitigated through techniques like RAG, fact-checking, and careful prompt design.

* Zhang et al. (2023). Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. https://arxiv.org/abs/2309.01219

## Few-Shot Learning
id: few-shot-learning
tags: prompting, techniques
level: intermediate

Few-shot learning is a technique where a small number of examples (typically 2-10) are provided in the prompt to demonstrate the desired task or output format. The LLM learns from these examples within the context window and applies the pattern to new inputs. This approach sits between zero-shot (no examples) and fine-tuning, offering a flexible way to guide model behavior without additional training.

* Brown et al. (2020). Language Models are Few-Shot Learners. https://arxiv.org/abs/2005.14165

## Chain-of-Thought (CoT)
id: chain-of-thought
tags: prompting, techniques, reasoning
level: intermediate

Chain-of-Thought prompting is a technique that encourages LLMs to break down complex reasoning tasks into intermediate steps before arriving at a final answer. By explicitly asking the model to "think step by step" or showing examples with reasoning traces, CoT improves performance on tasks requiring logical reasoning, mathematics, or multi-step problem solving.

* Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903
* Kojima et al. (2022). Large Language Models are Zero-Shot Reasoners. https://arxiv.org/abs/2205.11916

## Temperature
id: temperature
tags: fundamentals, parameters
level: intermediate

Temperature is a parameter that controls the randomness of an LLM's output. Lower temperatures (e.g., 0.0-0.3) make outputs more deterministic and focused on the most likely tokens, while higher temperatures (e.g., 0.7-1.0) increase diversity and creativity but may reduce coherence. Temperature is crucial for balancing between factual accuracy and creative generation depending on the use case.

* OpenAI (2023). API Reference: Temperature parameter. https://platform.openai.com/docs/api-reference/chat/create#temperature

## Embeddings
id: embeddings
tags: architecture, techniques
level: intermediate

Embeddings are numerical vector representations of text that capture semantic meaning in a high-dimensional space. Words, sentences, or documents with similar meanings have similar embedding vectors. LLMs use embeddings internally, and they're also used for tasks like semantic search, clustering, and RAG. Modern embedding models can represent entire paragraphs or documents as single vectors.

* Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781

## System Prompt
id: system-prompt
tags: prompting, techniques
level: basic

A system prompt is a special instruction given to an LLM at the beginning of a conversation that defines its role, behavior, and constraints. Unlike user prompts, system prompts typically persist across the entire conversation and set the overall context for how the model should respond. They're commonly used to specify expertise domains, output formats, or behavioral guidelines.

* Anthropic (2024). System Prompts Documentation. https://docs.anthropic.com/claude/docs/system-prompts

## GraphRAG
id: graphrag
tags: ai-engineering, techniques, new-paradigms-2024-25, wip
level: research

GraphRAG extends traditional RAG by organizing retrieved information into knowledge graphs rather than flat document collections. This approach captures relationships between entities and concepts, enabling more sophisticated reasoning about connections in the data. GraphRAG is particularly useful for complex domains like historical research, scientific literature, or legal documents where relationships between entities are crucial.

* Microsoft Research (2024). GraphRAG: Unlocking LLM discovery on narrative private data. https://www.microsoft.com/en-us/research/blog/graphrag/

## Transformer Architecture
id: transformer
tags: architecture, fundamentals
level: research

The transformer is the neural network architecture underlying modern LLMs, introduced in the "Attention Is All You Need" paper. It uses self-attention mechanisms to process input sequences in parallel rather than sequentially, enabling efficient training on large datasets. Key components include multi-head attention, positional encodings, and feed-forward layers. The transformer architecture revolutionized NLP and forms the basis of models like GPT, BERT, and Claude.

* Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762

## RLHF (Reinforcement Learning from Human Feedback)
id: rlhf
tags: training, alignment, safety
level: research

RLHF is a training technique used to align LLMs with human preferences and values. After initial pre-training, models are fine-tuned using feedback from human evaluators who rank different responses. This process helps models become more helpful, harmless, and honest. RLHF is crucial for creating safe, user-friendly AI assistants but introduces questions about whose values are being encoded.

* Ouyang et al. (2022). Training language models to follow instructions with human feedback. https://arxiv.org/abs/2203.02155
* Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback. https://arxiv.org/abs/2212.08073
