# Glossary: Generative AI and Language Models

## Scaling Laws
id: scaling-laws
en: Scaling Laws
tags: training, fundamentals
level: intermediate

In the field of artificial intelligence, scaling laws refer to empirically determined principles that describe the quantitative relationship between the performance of a neural network and its scaling dimensions. These mathematical power laws enable precise prediction of how much a model's prediction error decreases when the number of [[#Parameters]], the size of the training dataset, or the available compute is increased. They serve as a central instrument for resource planning, as they indicate the optimal ratio at which model size and data volume must grow together to maximize efficiency and avoid bottlenecks like overfitting before actual training begins.

* Kaplan, Jared, Sam McCandlish, Tom Henighan, et al. "Scaling Laws for Neural Language Models". arXiv:2001.08361. Preprint, arXiv, January 23, 2020. [https://doi.org/10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361).

## Stochastic Parrot
id: stochastic-parrot
en: Stochastic Parrot
tags: fundamentals, safety
level: basic

A **Stochastic Parrot** refers to [[#Large Language Model (LLM)|Large Language Models]] that generate text by stringing together linguistic forms based on statistical probabilities, without possessing actual understanding of meaning or communicative intent. The authors argue that these systems merely "parrot" patterns observed in vast training data and combine these sequences stochastically (randomly). Although the results often appear coherent and meaningful to human readers, this [[#Understanding|understanding]] is, according to the paper, an illusion, as the model has no connection to reality or the truth of what it says. While model capabilities have increased, the philosophical debate about semantics vs. syntax underlying this term remains current.

* Bender, Emily M., Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?" _Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency_ (New York, NY, USA), FAccT '21, Association for Computing Machinery, March 1, 2021, 610–23. [https://doi.org/10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922).

## Emergence in LLMs
id: emergenz
en: Emergence in LLMs
tags: fundamentals, training
level: advanced

The phenomenological appearance of complex capabilities (e.g., arithmetic, logical reasoning, Theory of Mind) in large models that were absent or only randomly present in smaller models of the same architecture. According to Wei et al. (2022), these abilities do not scale linearly but show a **phase transition**: performance remains near zero for a long time and jumps abruptly once a critical threshold of parameters and compute is reached.

Schaeffer et al. (2023) question this phenomenon as a possible "illusion" (**Mirage**). They argue that the observed suddenness primarily results from **discontinuous evaluation metrics** (e.g., *Exact Match*: "all or nothing"). When examining continuous metrics (e.g., token probabilities), performance improvement is often linear and predictable. Nevertheless, emergence remains relevant as a *user-side* experience: for practical applications, the transition from "useless" to "functional" often feels abrupt.

* Wei, Jason, et al. "Emergent Abilities of Large Language Models". _Transactions on Machine Learning Research_, 2022. [https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682).
* Schaeffer, Rylan, Brando Miranda, and Sanmi Koyejo. "Are Emergent Abilities of Large Language Models a Mirage?". _Advances in Neural Information Processing Systems_ (NeurIPS), Vol. 36, 2024. [https://arxiv.org/abs/2304.15004](https://arxiv.org/abs/2304.15004).

## In-Context Learning
id: in-context-learning
en: In-Context Learning
tags: prompting, fundamentals
level: intermediate

The ability of language models to solve tasks through instructions or examples (exemplars) in the prompt or [[#Context Window]] without any update to model weights (retraining/[[#Fine-Tuning]]). The term originated with Brown et al. (2020). Schulhoff et al. (2024) note, however, that the word "Learning" is misleading, as often no new abilities are learned. Instead, it is usually **Task Specification**: the model retrieves abilities or knowledge that were already latently present from [[#Pre-Training]] and uses the context merely for activation and alignment.

* Brown, Tom B., et al. "Language Models are Few-Shot Learners". _Advances in Neural Information Processing Systems_, Vol. 33, 2020.
* Schulhoff, Sander, et al. "The Prompt Report: A Systematic Survey of Prompt Engineering Techniques". arXiv:2406.06608. 2024 (esp. Section 2.2.1 and Appendix A.9).

## Inference
id: inferenz
en: Inference
tags: fundamentals, ai-engineering
level: basic

The process by which an already fully trained AI model is used to process new inputs and deliver results. In contrast to training (such as [[#Pre-Training]] or [[#Fine-Tuning]]), where the model learns and changes its internal connections, the model's knowledge remains static (frozen) during [[#Inference]].

* Pope, Reiner, et al. "Efficiently Scaling Transformer Inference". _Proceedings of Machine Learning and Systems_, Vol. 5, 2023. [https://arxiv.org/abs/2211.05102](https://arxiv.org/abs/2211.05102).

## Synthetic Data
id: synthetische-daten
en: Synthetic Data
tags: training, safety
level: intermediate

Synthetic data in this context refers to artificially generated teaching materials specifically created to mimic the didactic clarity and structure of high-quality textbooks ("Textbooks Are All You Need"). Instead of using unstructured or error-prone information from the internet, these AI-generated texts and exercises serve to precisely convey logical connections and algorithmic thinking. While high-quality synthetic data can improve "reasoning," its unfiltered or exclusive use in recursive training loops poses significant risks to model quality (see **[[#Model Collapse]]**).

* Gunasekar, Suriya, Yi Zhang, Jyoti Aneja, et al. "Textbooks Are All You Need". arXiv:2306.11644. Preprint, arXiv, October 2, 2023. [https://doi.org/10.48550/arXiv.2306.11644](https://doi.org/10.48550/arXiv.2306.11644).

## AI Engineering
id: ai-engineering
en: AI Engineering
tags: ai-engineering, fundamentals
level: basic

An interdisciplinary field that combines methods from systems engineering, software engineering, computer science, and human-centered design to develop, deploy, and maintain AI systems. Unlike pure model development, AI Engineering encompasses the entire lifecycle—from prototype to production. The focus is on creating robust, scalable, and trustworthy systems that reliably solve real problems and are aligned with human needs and operational goals—especially in high-stakes environments.

* Carnegie Mellon Software Engineering Institute - AI Engineering Current (2025). https://www.sei.cmu.edu/artificial-intelligence-engineering/
* MIT Professional Education - What is Artificial Intelligence Engineering? October 2, 2023. https://professionalprograms.mit.edu/blog/technology/artificial-intelligence-engineering/
* CMU Course - Machine Learning in Production / AI Engineering Spring 2025. https://mlip-cmu.github.io/s2025/

## System Prompt
id: system-prompt
en: System Prompt
tags: prompting, ai-engineering
level: basic

A system prompt functions as an initial configuration layer that provides an [[#Large Language Model (LLM)|LLM]] with overarching behavioral instructions and restrictions before user interaction. In practice, this mechanism serves to inject dynamic context—such as the current date—and technically enforce specific output standards, such as code formatting. Since these instructions exist independently of model weights, they are treated as flexible components that are iteratively optimized through continuous versioning to precisely control the model's response behavior without requiring retraining.

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary

## Custom Instruction
id: custom-instruction
en: Custom Instruction
tags: prompting
level: basic

A feature (made popular by ChatGPT) that functions as a persistent "mini [[#System Prompt]]" at the user level. It allows users to store permanent contextual information (e.g., "I am a Python developer") and preferences (e.g., "Always respond concisely without filler words") that are automatically prepended to every new conversation.

* ChatGPT Custom Instructions. https://help.openai.com/en/articles/8096356-chatgpt-custom-instructions

## Fine-Tuning
id: fine-tuning
en: Fine-Tuning
tags: training
level: intermediate

The process of continuing to train an already pre-trained language model ([[#Pre-Training|Pre-trained Model]]) with a specific, smaller dataset. While pre-training builds broad knowledge and language understanding, fine-tuning serves to specialize the model for specific tasks (e.g., coding, medical analysis) or a particular writing style. It adapts the weights ([[#Parameters]]) so that the model imitates the patterns of the new dataset. Anthropic notes that models without this step (Bare Models) often struggle to follow instructions, as they are merely trained to predict text, not to act as helpful assistants.

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary

## Red Teaming
id: red-teaming
en: Red Teaming
tags: safety, evaluation
level: intermediate

Structured adversarial testing of AI systems in which individuals attempt to provoke harmful, undesirable, or erroneous outputs to identify, measure, and reduce vulnerabilities. Unlike traditional cybersecurity red teaming, it encompasses both security-related and content-related risks such as [[#Bias]], misinformation, or toxic content. The goal is to identify weaknesses _before_ release to make the model more robust against [[#Jailbreak]] and [[#Prompt Injection]].

* Ganguli, Deep, Liane Lovitt, Jackson Kernion, et al. "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned". arXiv:2209.07858. Preprint, arXiv, November 22, 2022. [https://doi.org/10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858).
* Video: [Prompt Injection, Jailbreaking, and Red Teaming – Sander Schulhoff](https://youtu.be/J9982NLmTXg)

## Prompt Engineering
id: prompt-engineering
en: Prompt Engineering
tags: prompting
level: basic

Prompt Engineering is a systematic and iterative process for developing and optimizing input prompts, where the prompting technique used is modified or changed to effectively guide Large Language Models (LLMs) and maximize the quality of generated outputs for specific tasks.

* Schulhoff, Sander, Michael Ilie, Nishant Balepur, et al. "The Prompt Report: A Systematic Survey of Prompt Engineering Techniques". arXiv:2406.06608. Preprint, arXiv, February 26, 2025. [https://doi.org/10.48550/arXiv.2406.06608](https://doi.org/10.48550/arXiv.2406.06608).

## Zero-Shot and Few-Shot
id: zero-shot-few-shot
en: Zero-Shot & Few-Shot Learning
tags: prompting
level: basic

This is not a training phase where weights ([[#Parameters]]) are updated, but a form of [[#In-Context Learning]] where the model is conditioned "at runtime" exclusively through input in the [[#Context Window]]. While Zero-Shot uses only a natural language instruction without examples and One-Shot provides exactly one reference example, Few-Shot fills the context with as many demonstrations as possible (typically 10 to 100) to provide the model with the desired pattern.

* Brown, Tom B., et al. "Language Models are Few-Shot Learners". *Advances in Neural Information Processing Systems*, Vol. 33, 2020, pp. 1877–1901. [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165).

## Grokking
id: grokking
en: Grokking
tags: training
level: advanced

Grokking refers to a counterintuitive phenomenon in neural network training where generalization ability (understanding of new data) only sets in abruptly long after the model has already perfectly memorized the training data (overfitting). While classical teachings recommend stopping training as soon as the model begins to merely memorize training data, Power et al. (2022) show that with extremely prolonged optimization, a transition can suddenly occur: the model discards the "memorized" complex solution and finds the simpler, true rule (the algorithm) behind the data. This suggests that "understanding" is often harder to find in the solution space than memorization and requires patience during training.

* Power, Alethea, et al. "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets". _arXiv preprint_, 2022. [https://arxiv.org/abs/2201.02177](https://arxiv.org/abs/2201.02177).
* Video: [TODO_TITLE](https://youtu.be/D8GOeCFFby4)

## Many-Shot
id: many-shot
en: Many-Shot Learning
tags: prompting
level: intermediate

An evolution of [[#Zero-Shot and Few-Shot|Few-Shot Learning]] made possible by extremely large [[#Context Window|context windows]] (e.g., 1 million [[#Token|tokens]] with Gemini). Instead of giving only 5 or 10 examples, the model is presented with hundreds or thousands of examples (which can encompass entire datasets) in the prompt. Studies show that this often yields better results than traditional [[#Fine-Tuning]], as the model analyzes patterns directly in the context window.

* Agarwal, Rishabh, Avi Singh, Lei M. Zhang, et al. "Many-Shot In-Context Learning". arXiv:2404.11018. Preprint, arXiv, April 16, 2024. [http://arxiv.org/abs/2404.11018](http://arxiv.org/abs/2404.11018).

## Tree of Thoughts (ToT)
id: tree-of-thoughts
en: Tree of Thoughts
tags: prompting
level: advanced

A framework for problem-solving that originally went beyond simple prompting by combining an external **search algorithm** (like breadth-first or depth-first search) with an LLM. The model generates multiple possible solution steps ("thoughts"), which are evaluated by the model itself and selected by the algorithm. Unlike [[#Chain of Thought (CoT)|Chain-of-Thought]] (a linear pass), ToT enables active exploration, evaluation, and rejection of solution paths (_Backtracking_). In practice, the concept is now also adapted as a pure **prompting technique**, where the model is instructed to simulate this exploration and evaluation process within a single output.

* Yao, Shunyu, et al. "Tree of Thoughts: Deliberate Problem Solving with Large Language Models". arXiv:2305.10601. 2023. [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601).

## AI Slop
id: ai-slop
en: AI Slop
tags: safety
level: basic

A pejorative term (analogous to "spam" for emails) for mass-generated, low-quality AI content flooding the internet. Slop is characterized by appearing superficially like useful content (often SEO-optimized) but being redundant, imprecise, or meaningless in substance. Unlike [[#Hallucinations (Confabulations)|hallucinations]] (which are errors), slop is the result of careless or malicious use of LLMs merely to generate attention without added value.

* Simon Willison. "Slop is the new name for unwanted AI-generated content". _Simon Willison's Weblog_, May 8, 2024. [https://simonwillison.net/2024/May/8/slop/](https://simonwillison.net/2024/May/8/slop/).

## Latent Space
id: latent-space
en: Latent Space
tags: architecture, fundamentals
level: advanced

The high-dimensional, abstract vector space in which a model represents information. While we see text or pixels, the model "thinks" in coordinates within this space. Concepts that are semantically similar lie spatially close together in this space (see [[#Embedding]]). Understanding and manipulating this space is central to [[#Mechanistic Interpretability]] and explains why models can form analogies: they perform computational operations (vector arithmetic) with meanings.

* Liu, Ziming, et al. "Physics of Language Models: Part 1, Context-Free Grammar". _arXiv preprint_, 2023. [https://arxiv.org/abs/2305.13673](https://arxiv.org/abs/2305.13673).

## LLM Council
id: llm-council
en: LLM Council
tags: agents, ai-engineering
level: advanced

An architecture within [[#Multi-Agent Systems]] where a group of different language models (or different personas of the same model) work together on a task, rather than a single model generating an isolated answer. Similar to a human expert panel, the "council" members independently generate solution proposals, critique each other (peer review), and then consolidate the results into a final answer. This approach leverages the "wisdom of the crowd" (ensemble learning) to reduce [[#Hallucinations (Confabulations)]] and balance bias, as errors from a single model can be corrected by the majority.

* https://lmcouncil.ai

## Shadow AI
id: shadow-ai
en: Shadow AI
tags: safety, ai-engineering
level: basic

The phenomenon where employees in companies independently use AI tools (like ChatGPT or DeepL) for work tasks without IT department knowledge or approval. This is one of the biggest current risks for organizations (data leakage), as sensitive company data often unknowingly ends up in the training data of public models.

## Open Weights
id: open-weights
en: Open Weights (vs. Open Source)
tags: fundamentals
level: basic

An important nuance in the licensing debate. "Open Source" classically means that training data, code, and instructions are freely available. However, many modern "open" models (like Llama from Meta or Mistral) are only **Open Weights**. This means: you get the fully trained model (the weights) for free use, but the manufacturer keeps secret _what_ exactly it was trained on (the "recipe"). This is important for questions about copyright and transparency.

* Liesenfeld, A., & Dingemanse, M. (2024). Rethinking open source generative AI: Open-washing and the EU AI Act. Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 1774–1787. [https://doi.org/10.1145/3630106.3659005](https://doi.org/10.1145/3630106.3659005).

## Frontier Model
id: frontier-model
en: Frontier Model
tags: fundamentals
level: basic

Refers to the absolute cutting edge of AI development at any given time. Frontier Models are those models that push the current boundaries of what AI can do.

## LLM-as-a-Judge
id: llm-as-judge
en: LLM-as-a-Judge
tags: evaluation
level: intermediate

An evaluation method where a strong LLM (e.g., GPT-4) is used to assess the responses of other (often smaller or more specialized) models. Since human evaluation is expensive and slow, and static [[#Benchmark|benchmarks]] are often distorted by _data contamination_, the strong model acts as a juror, grading aspects like relevance, coherence, and helpfulness. Critics like Zheng et al. point out the **self-preference bias**—the tendency of models to prefer answers generated by themselves or similar models.

* Zheng, Lianmin, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". _Advances in Neural Information Processing Systems_, Vol. 36, 2024. [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685).

## Chain of Thought (CoT)
id: chain-of-thought
en: Chain of Thought
tags: prompting
level: intermediate

A prompting technique that causes Large Language Models (LLMs) to decompose complex tasks into a sequence of intermediate, natural language reasoning steps ("thought chain") before generating the final answer. This method, which according to Wei et al. (2022) only appears effectively as an [[#Emergence in LLMs|emergent ability]] in sufficiently large models, enables significant performance improvements on mathematical and reasoning problems by emulating human problem-solving processes. Technically, however, this is not formal symbolic logic but a probabilistic simulation of argumentation patterns, which is why the generated steps may appear coherent but can be susceptible to logical hallucinations ("Unfaithful Reasoning").

* Wei, Jason, et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". *Advances in Neural Information Processing Systems*, Vol. 35, 2022. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903).

## Context Engineering
id: context-engineering
en: Context Engineering
tags: prompting, ai-engineering
level: intermediate

The systematic design and optimization of the information context for LLMs, with the goal of maximizing the quality and reliability of model responses under limited resources. It encompasses strategies for selecting, compressing, and arranging information in the context window.

* Mei, Lingrui, Jiayu Yao, Yuyao Ge, et al. "A Survey of Context Engineering for Large Language Models". arXiv:2507.13334. Preprint, arXiv, July 21, 2025. [https://doi.org/10.48550/arXiv.2507.13334](https://doi.org/10.48550/arXiv.2507.13334).

## Context Window
id: context-window
en: Context Window
tags: architecture, fundamentals
level: basic

The maximum number of [[#Token|tokens]] that a model can process in a single pass. Technically, this is the area that the [[#Attention (Self-Attention)|Self-Attention]] mechanism can access. It functions as the model's "working memory." Unlike the knowledge "hardwired" in the neural network from training, the context window is ephemeral and exists only for the duration of the interaction. A larger window enables processing entire books or long code bases in a single prompt.

* Vaswani, Ashish, et al. "Attention Is All You Need". *Advances in Neural Information Processing Systems*, Vol. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

## Context Rot
id: context-rot
en: Context Rot
tags: architecture
level: intermediate

A phenomenon where LLM performance degrades with increasing input context length and decreasing information density. Unstructured accompanying text acts as noise that distracts attention from relevant instructions.

* Hong, Kelly, Anton Troynikov, and Jeff Huber. Context Rot: How Increasing Input Tokens Impacts LLM Performance. Chroma, 2025. [https://research.trychroma.com/context-rot](https://research.trychroma.com/context-rot).

## Lost-in-the-Middle
id: lost-in-the-middle
en: Lost-in-the-Middle
tags: architecture
level: intermediate

An observation that language models can retrieve and process information at the beginning (primacy effect) and end (recency effect) of the [[#Context Window|context window]] significantly better than information placed in the middle of long contexts.

* Liu, Nelson F., et al. "Lost in the Middle: How Language Models Use Long Contexts". _Transactions of the Association for Computational Linguistics_, Vol. 12, 2024, pp. 157–73. [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172).

## Bias
id: bias
en: Bias
tags: safety
level: intermediate

In the context of Large Language Models (LLMs), bias refers to systematic unequal treatment or outcomes between social groups that result from historical and structural power asymmetries and should not be understood as mere statistical errors. Gallegos et al. (2024) differentiate between representational harms (such as stereotyping, derogatory language, or invisibilization through exclusionary norms) and allocative harms (unequal allocation of resources or opportunities). These distortions arise throughout the entire lifecycle—from imbalanced training data to model design decisions to inadequate evaluation metrics—and stand in direct tension with fairness concepts like Group Fairness (statistical parity between groups) and Individual Fairness.

* Gallegos, Isabel O., Ryan A. Rossi, Joe Barrow, et al. "Bias and Fairness in Large Language Models: A Survey". _Computational Linguistics_ 50, No. 3 (2024): 1097–179. [https://doi.org/10.1162/coli_a_00524](https://doi.org/10.1162/coli_a_00524).

## Benchmark
id: benchmark
en: Benchmark
tags: evaluation
level: basic

Standardized test sets for evaluating LLM performance across various disciplines (e.g., logic, code, general knowledge). A central methodological problem is *data contamination*, where test questions were already contained in the training dataset, which distorts results ("memorizing" instead of "reasoning").

* Liang, Percy, et al. "Holistic Evaluation of Language Models". *Annals of the New York Academy of Sciences*, 2023. [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110).

## Sycophancy
id: sycophancy
en: Sycophancy
tags: safety
level: intermediate

Sycophancy in Large Language Models refers to the tendency of models to excessively agree with or flatter users, where this prioritization of user satisfaction often comes at the expense of factual accuracy and ethical principles; this behavior manifests specifically in models providing inaccurate information to meet user expectations, giving unethical advice when prompted, or failing to correct false premises in user queries.

* Malmqvist, Lars. "Sycophancy in Large Language Models: Causes and Mitigations". Preprint, November 22, 2024. [https://arxiv.org/abs/2411.15287v1](https://arxiv.org/abs/2411.15287v1).
* Video: [TODO_TITLE](https://youtu.be/nvbq39yVYRk)

## Attention (Self-Attention)
id: attention
en: Self-Attention
tags: architecture
level: advanced

A mechanism that enables neural networks to model relationships between words ([[#Token|tokens]]) in a sequence, regardless of how far apart they are. Instead of rigidly reading text sequentially, the mechanism calculates a weighting (relevance) for each word that indicates how strongly it is connected to every other word in the context. Formally, this is described as mapping a query to a set of key-value pairs.

* Vaswani, Ashish, et al. "Attention Is All You Need". *Advances in Neural Information Processing Systems*, Vol. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

## Transformer
id: transformer
en: Transformer
tags: architecture
level: advanced

A network architecture introduced by Google Brain in 2017 that completely dispenses with recurrence (loops) and convolutions, instead relying exclusively on [[#Attention (Self-Attention)|Attention]] mechanisms. This architecture made it possible for the first time to massively parallelize language models, which drastically reduced training times and enabled training on gigantic datasets (and thus the development of today's [[#Large Language Model (LLM)|LLMs]] like GPT or BERT). A transformer typically consists of an encoder and a decoder stack (though models like GPT use only the decoder part).

* Vaswani, Ashish, et al. "Attention Is All You Need". *Advances in Neural Information Processing Systems*, Vol. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

## Mechanistic Interpretability
id: mechanistic-interpretability
en: Mechanistic Interpretability
tags: safety, architecture
level: advanced

Mechanistic Interpretability refers to the methodological approach of fully understanding the complex internal computations of AI models like [[#Transformer|Transformers]] through reverse engineering, similar to translating incomprehensible machine code back into human-readable source code. The goal is not merely to observe the model from outside but to identify concrete algorithmic patterns and mechanical circuits within the weights, thereby gaining a better understanding of how behaviors and potential safety risks emerge.

* Anthropic. A Mathematical Framework for Transformer Circuits. https://transformer-circuits.pub/2021/framework/index.html

## Reinforcement Learning
id: reinforcement-learning
en: Reinforcement Learning
tags: training
level: intermediate

A subfield of machine learning where an agent learns to make decisions by performing actions in an environment and receiving positive or negative feedback (reward). In the context of LLMs (see [[#Reinforcement Learning from Human Feedback (RLHF)|RLHF]]), RL does not serve to learn language (that happens in [[#Pre-Training]]) but to optimize behavioral strategies to align generated text with human preferences.

* Sutton, Richard S., and Andrew G. Barto. _Reinforcement Learning: An Introduction_. 2nd ed., MIT Press, 2018. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html).
* Video: [TODO_TITLE](https://youtu.be/LHsgtcNNM0A)

## Temperature
id: temperature
en: Temperature
tags: fundamentals, ai-engineering
level: basic

**Temperature** is a crucial hyperparameter for controlling randomness in generating the next [[#Token]], with Andrej Karpathy describing this inference process not as deterministic computation but as "flipping a weighted coin" (sampling). While extremely low values (near 0) lead to deterministic "greedy decoding," where the model rigidly chooses the most likely word and tends toward repetition, high values flatten the probability curve. This gives less likely terms a chance, which promotes creativity but also increases the risk of [[#Hallucinations (Confabulations)|hallucinations]]. Technically, it directly intervenes in the [[#Logits & Softmax|logits]].

* Karpathy, Andrej. "Intro to Large Language Models". _YouTube_, 2023. [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g).

## Logits & Softmax
id: logits-softmax
en: Logits & Softmax
tags: architecture
level: advanced

The raw, unnormalized numerical values that the neural network produces as the very last step before output. For each word in the vocabulary, there is a logit value; the higher the value, the more "fitting" the model finds the word. Since these numbers are hard to interpret (e.g., minus infinity to plus infinity), they are converted by the **softmax function** into probabilities (between 0 and 1) that sum to 100%. [[#Temperature]] intervenes precisely in this step by scaling (smoothing or sharpening) the logits before the softmax calculation.

## Vibe Checking
id: vibe-checking
en: Vibe Check
tags: evaluation
level: intermediate

**VibeCheck** is a scientific framework (presented by Dunlap et al. at ICLR 2025) that automatically identifies and quantifies subjective and hard-to-grasp properties of language models—so-called "vibes" like tone, formatting, or humor—to explain the discrepancy between static [[#Benchmark|benchmarks]] and human preference. A valid "vibe" is formally defined as an axis that is **well-defined** (consensus among evaluators), **differentiating** (reliably distinguishes models), and **user-oriented** (correlates with human preferences), thereby translating informal impressions like "feels smarter" into measurable data.

* Dunlap, Lisa, Krishna Mandal, Trevor Darrell, Jacob Steinhardt, and Joseph E. Gonzalez. "VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models". arXiv:2410.12851. Preprint, arXiv, April 19, 2025. [https://doi.org/10.48550/arXiv.2410.12851](https://doi.org/10.48550/arXiv.2410.12851).

## Pre-Training
id: pre-training
en: Pre-Training
tags: training
level: basic

The first development phase in which a neural network is trained on large amounts of internet text. The primary computational task is predicting the statistically most likely next [[#Token]] (word part) in a sequence ([[#Next Token Prediction]]). Karpathy compares this process to lossy data compression. The result is a **Base Model**: a system that can complete text documents but has no specific orientation toward dialogues or assistance tasks, merely reproducing the patterns of the training data.

* [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)

## Post-Training
id: post-training
en: Post-Training
tags: training
level: intermediate

The downstream adaptation process through which the base model is aligned for interaction as an assistant. This step uses [[#Fine-Tuning|Supervised Fine-Tuning (SFT)]] to condition the model on datasets of question-answer pairs (imitation of human specifications). Additionally, [[#Reinforcement Learning]] (RL) is applied to optimize response behavior through reward mechanisms and—in newer models—to establish internal processing steps ([[#Chain of Thought (CoT)|Chain of Thought]]) for error correction. The goal is to transform the model from pure text completion to instruction-compliant response behavior.

* [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)

## Reinforcement Learning from Human Feedback (RLHF)
id: rlhf
en: RLHF
tags: training, safety
level: intermediate

A fine-tuning method used to align models with human values and intentions (alignment). The process consists of three steps: 1. Collecting human comparison data (which of two answers is better?), 2. Training a reward model that predicts this human preference, and 3. Optimizing the language model using reinforcement learning (usually PPO - Proximal Policy Optimization) against this reward model. RLHF was the key factor that transformed models like GPT-3 into user-friendly assistants like ChatGPT.

* Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences". _Advances in Neural Information Processing Systems_, Vol. 30, 2017. [https://arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741).
* Stiennon, Nisan, et al. "Learning to summarize with human feedback". _Advances in Neural Information Processing Systems_, Vol. 33, 2020. [https://arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325).

## Constitutional AI
id: constitutional-ai
en: Constitutional AI
tags: training, safety
level: advanced

Constitutional AI (CAI) is a training method developed by Anthropic where an AI system is primarily guided by a list of rules or principles written in natural language—a so-called "constitution"—rather than relying on human labels to identify harmful outputs. The process occurs in two phases: First, in a supervised learning phase, the model uses the constitution to critique and revise its own responses (self-improvement); then, in a [[#Reinforcement Learning]] phase (RLAIF), a preference model is used that is based on AI feedback (rather than human feedback) to train the model to give harmless, transparent, and helpful answers without being evasive.

* Bai, Yuntao, Saurav Kadavath, Sandipan Kundu, et al. "Constitutional AI: Harmlessness from AI Feedback". arXiv:2212.08073. Preprint, arXiv, December 15, 2022. [https://doi.org/10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073).
* Claude's Constitution. https://www.anthropic.com/news/claudes-constitution

## Character (Persona)
id: character
en: Character / Persona
tags: safety, training
level: intermediate

The term for the totality of specific behavioral dispositions and personality traits deliberately imparted to a [[#Large Language Model (LLM)|Large Language Model]] (LLM) during the training process—especially in the [[#Alignment]] phase. Unlike pure safety mechanisms that primarily aim to avoid harmful outputs, character development serves to establish positive attributes such as curiosity, nuance, openness to diverse perspectives, and honesty. A defined character should enable the model to act consistently in ethically complex situations, deal transparently with its own identity as an artificial system, and engage in constructive dialogue without feigning artificial neutrality or uncritically confirming user views.

* Claude's Character. Anthropic. https://www.anthropic.com/research/claude-character

## Alignment
id: alignment
en: AI Alignment
tags: safety, training
level: intermediate

Alignment is defined as the orientation of AI systems toward complex human intentions and values, concretely operationalized through the principles "helpful, honest, and harmless" (HHH). Since these qualitative goals cannot be specified through handwritten rules or simple mathematical functions, alignment is technically solved as a "preference modeling" problem: the system learns not through mere imitation of data but through iterative human feedback (comparisons of action options) to approximate an internal reward function. This method bridges the communication gap between vague human intent and machine optimization by ensuring that the model learns even nuanced, hard-to-define safety and utility standards in a scalable way that would not be robustly representable through pure supervised learning.

* Askell, Amanda, et al. "A General Language Assistant as a Laboratory for Alignment". *Anthropic*, 2021. [https://arxiv.org/abs/2112.00861](https://arxiv.org/abs/2112.00861).

## Large Language Model (LLM)
id: llm
en: Large Language Model
tags: fundamentals
level: basic

A probabilistic model based on neural networks that has been trained on vast amounts of text to learn statistical patterns of language. It is characterized by a high parameter count (billions to trillions) and emergent abilities that go beyond pure language modeling (e.g., logical reasoning).

* Zhao, Wayne Xin, et al. "A Survey of Large Language Models". _arXiv preprint_, 2023. [https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223).

## Token
id: token
en: Token
tags: fundamentals, architecture
level: basic

The atomic unit of information processing in a language model, aptly described by Karpathy as the "atoms" of the system. Instead of linguistically dividing text into syllables or words, a deterministic algorithm (usually Byte Pair Encoding) breaks the input into statistically frequent fragments that are processed as a sequence of integers. For the model, there is no text, only this sequence of numbers, with one [[#Token]] corresponding to approximately 4 characters or 0.75 words in English. Since the model perceives these tokens as indivisible units, this architecture also explains why LLMs paradoxically often fail at simple tasks like spelling or character counting—they "see" the word as a whole block and not the individual letters within it.

* Karpathy, Andrej. "Let's build the GPT Tokenizer". _YouTube_, January 17, 2024. [https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE).

## Tokenizer
id: tokenizer
en: Tokenizer
tags: fundamentals, architecture
level: basic

A deterministic algorithm that functions as a translator to convert human-readable text into a sequence of integers, since neural networks cannot process letters but only mathematical values. Technically, this is usually based on **Byte Pair Encoding (BPE)**: the tokenizer analyzes raw bytes and iteratively merges the most frequent character pairs into new, larger units until a fixed vocabulary is reached. These [[#Token|tokens]] form the indivisible "atoms" of the model; since the model can no longer break down the content of a token (e.g., a word part) into individual letters, the tokenizer is often the hidden cause of problems with tasks like spelling or character counting.

* Karpathy, Andrej. "Let's build the GPT Tokenizer". _YouTube_, January 17, 2024. [https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE).

## Embedding
id: embedding
en: Embedding
tags: architecture, ai-engineering
level: intermediate

A mathematical representation of [[#Token|tokens]] or text passages as vectors in a high-dimensional space. In this space, semantically similar concepts (e.g., "king" and "emperor") lie geometrically close together, allowing the model to compute semantic relationships.

* Mikolov, Tomas, et al. "Efficient Estimation of Word Representations in Vector Space". _arXiv preprint_, 2013. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781).

## Next Token Prediction
id: next-token-prediction
en: Next Token Prediction
tags: fundamentals, training
level: basic

Next Token Prediction refers to the fundamental operating principle of autoregressive language models, where based on a sequence of preceding [[#Token|tokens]], the probability distribution for the immediately following token is determined ($P(w_t | w_{1:t-1})$). This probabilistic method serves both in pre-training as a learning task to capture linguistic and content patterns and during inference for the step-by-step generation of new texts.

* Bengio, Yoshua, et al. "A Neural Probabilistic Language Model". _Journal of Machine Learning Research_, Vol. 3, 2003, pp. 1137–1155. [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

## Retrieval Augmented Generation (RAG)
id: rag
en: RAG (Retrieval Augmented Generation)
tags: ai-engineering
level: intermediate

RAG is an approach that couples generative language models with an external information retrieval system (retriever). In this method, the model generates answers not exclusively from internal parameters stored during training but first retrieves relevant documents from an external knowledge base (e.g., a vector database). These retrieved text passages are fed into the generation process as additional context. This method can increase the factual accuracy of generated texts and reduce the tendency for hallucinations. Additionally, the approach enables updating available knowledge by simply exchanging the document index without retraining the neural network.

* Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". _Advances in Neural Information Processing Systems_, Vol. 33, 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401).

## Vector Database
id: vektordatenbank
en: Vector Database
tags: ai-engineering
level: intermediate

A specialized database that stores information not as text or tables but as high-dimensional vectors (embeddings). It enables semantic search: instead of searching for exact keywords, the database calculates the mathematical distance (e.g., cosine similarity) between the query vector and stored document vectors. Efficiently searching these high-dimensional spaces (_similarity search_) requires specialized indexing structures to remain performant even with billions of records. This forms the technological foundation for RAG systems, as it enables rapid retrieval of content-relevant context from vast data volumes.

* Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity search with GPUs". _IEEE Transactions on Big Data_ 7, No. 3 (2019): 535–47. [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734).

## Tool Use and Function Calling
id: tool-use
en: Tool Use / Function Calling
tags: agents, ai-engineering
level: intermediate

The ability of a model to recognize that a request requires external tools (e.g., calculator, weather API, database query) and then generate structured commands (usually JSON) that can be executed by a software environment. The result of the execution is returned to the model to formulate the final answer.

* Schick, Timo, and Jane Dwivedi-Yu. "Toolformer: Language Models Can Teach Themselves to Use Tools". _Advances in Neural Information Processing Systems_, Vol. 36, 2023. [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761).

## Model Context Protocol (MCP)
id: mcp
en: Model Context Protocol
tags: ai-engineering, agents
level: intermediate

An open standard that serves as a universal interface to securely and seamlessly connect AI assistants with external data sources—such as content repositories, business tools, and development environments. Instead of having to develop an individual, fragmented integration for each system, MCP offers a standardized architecture through which AI models gain direct access to relevant, isolated data to deliver more precise and context-aware answers.

* Anthropic. "Introducing the Model Context Protocol". _Anthropic News_, November 25, 2024. [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol).

## AI Agent
id: ai-agent
en: AI Agent
tags: agents
level: intermediate

An autonomous system that perceives its environment and proactively acts to achieve defined goals. Unlike passive models, an agent uses an LLM as a central reasoning unit to create multi-step plans and use external tools or APIs for execution. The core process is a continuous loop of observation, decision, and action.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, and Manoj Karkee. "AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (September 2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).

## Agentic AI
id: agentic-ai
en: Agentic AI
tags: agents
level: intermediate

A paradigm in AI development that describes the degree of action autonomy (_agency_) of a system. It refers to the transition from generative AI, which merely creates content, to systems that function as active problem solvers. Agentic AI is characterized by the ability to independently decompose complex tasks into sub-steps, verify its own results (self-reflection), and dynamically adapt the solution path when errors occur.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, and Manoj Karkee. "AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (September 2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).

## Multi-Agent Systems
id: multi-agent-systems
en: Multi-Agent Systems
tags: agents
level: advanced

Systems in which multiple specialized AI agents interact with each other (cooperate, debate, or compete) to solve complex problems. Through role division (e.g., a coder, a reviewer), often better results can be achieved than with a single, monolithic agent.

* Li, Guohao, et al. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society". _Advances in Neural Information Processing Systems_, Vol. 36, 2023. [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760).

## World Models
id: world-models
en: World Models
tags: agents, fundamentals
level: advanced

A World Model refers to a generative AI system that learns a compressed and abstract representation of its physical environment to precisely predict its dynamics and future states. Technically, this concept is usually realized through a visual component for data reduction into a latent space and a temporal component for simulating future events based on one's own actions. This architecture enables an agent or robot to mentally simulate potential action consequences and design complex plans without having to risky try every step in the real world. It thus functions as an internal simulator that replaces mere reaction to stimuli with anticipatory planning and gives machine systems a functional intuition for causality and physical laws.

* Ha, David, and Jürgen Schmidhuber. "World Models". _arXiv preprint_, 2018. [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

## Hallucinations (Confabulations)
id: halluzinationen
en: Hallucinations / Confabulations
tags: safety, fundamentals
level: basic

The generation of content that sounds grammatically and semantically plausible but is factually incorrect or not based on the training data/sources. The term "confabulation" is increasingly preferred (e.g., by Geoffrey Hinton) as it more accurately describes the process of "filling in gaps" without reference to reality than a perceptual disorder would.

* Ji, Ziwei, et al. "Survey of Hallucination in Natural Language Generation". _ACM Computing Surveys_, Vol. 55, No. 12, 2023. [https://arxiv.org/abs/2202.03629](https://arxiv.org/abs/2202.03629).

## Understanding
id: understanding
en: Understanding
tags: fundamentals, safety
level: advanced

A highly contested term in AI research. While LLMs show high _functional competence_ (output is correct), critics dispute that they possess _formal competence_ (understanding of meaning/semantics). It is often argued that models are merely statistical parrots that manipulate forms without grasping their content.

* Bender, Emily M., and Alexander Koller. "Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data". _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, 2020. [https://aclanthology.org/2020.acl-main.463/](https://aclanthology.org/2020.acl-main.463/).

## Consciousness and LLMs
id: bewusstsein
en: Consciousness in LLMs
tags: fundamentals, safety
level: advanced

The debate about whether language models possess subjective experience (_Subjective Experience_ or _Sentience_). Philosopher David Chalmers analyzes this using necessary indicators (_Feature X_) that current models lack. He argues that current LLMs most likely lack consciousness because they are primarily **feed-forward systems** without memory loops (**recurrence**) and possess no robust **self-models** (internal monitoring) or **unified agency**. However, Chalmers sketches a roadmap to **LLM+** (extended multimodal systems), where through technical additions like a _Global Workspace Architecture_ or embodied interaction (_Embodiment_) in virtual worlds, genuine consciousness candidates could emerge.

* Chalmers, David J. "Could a Large Language Model be Conscious?" arXiv:2303.07103. Preprint, arXiv, August 18, 2024. [https://doi.org/10.48550/arXiv.2303.07103](https://doi.org/10.48550/arXiv.2303.07103).

## AGI (Artificial General Intelligence)
id: agi
en: AGI (Artificial General Intelligence)
tags: fundamentals
level: intermediate

The term is controversial and not uniformly defined. According to Bennett (2025), AGI is not a pure software construct but a physically anchored complete system (hardware and software) that possesses the autonomy and adaptability of an "artificial scientist." Instead of defining intelligence merely through human performance on known tasks, Bennett defines AGI as the ability to achieve goals in a broad range of unknown environments, where the true measure is not raw compute but **sample efficiency** (learning from few data) and **energy efficiency** in adapting to new problems.

* Bennett, Michael Timothy. _What the F*ck Is Artificial General Intelligence?_ Vol. 16057. 2026. [https://doi.org/10.1007/978-3-032-00686-8_4](https://doi.org/10.1007/978-3-032-00686-8_4).

## Parameters
id: parameter
en: Parameters
tags: fundamentals, architecture
level: basic

The internal configuration variables (primarily weights and biases) of a neural network that are learned and adjusted during the training process through mathematical optimization (backpropagation) to minimize prediction error. They represent the strength of connections between artificial neurons and thus store all extracted knowledge and capabilities of the model in the form of gigantic number matrices. During [[#Inference]], these values remain static; the sheer number of parameters (often measured in billions) is considered, according to [[#Scaling Laws]], the primary indicator of a model's potential capacity, while simultaneously directly determining the need for compute and graphics memory (VRAM).

* Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. _Deep Learning_. MIT Press, 2016. [http://www.deeplearningbook.org](http://www.deeplearningbook.org/).

## Mixture of Experts (MoE)
id: moe
en: Mixture of Experts
tags: architecture
level: advanced

A model architecture where the neural network is not activated as a single monolithic block but is divided into many small sub-networks ("experts"). For each token, a "router" decides which experts (usually only 1-2) are activated. This allows models with extremely many parameters (e.g., GPT-4, Mixtral) that are nevertheless fast and cost-effective in [[#Inference]], since only a fraction of the network computes at any time.

* Fedus, William, Barret Zoph, and Noam Shazeer. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". _Journal of Machine Learning Research_, Vol. 23, 2022. [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961).

## Quantization
id: quantisierung
en: Quantization
tags: ai-engineering, architecture
level: intermediate

A technique to reduce the memory footprint and computational load of an LLM by reducing the precision of model weights (e.g., from 16-bit floating-point numbers to 4-bit integers). This makes it possible to run huge models on consumer hardware (local laptops/GPUs), often with only minimal quality loss.

* Dettmers, Tim, et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale". _Advances in Neural Information Processing Systems_, Vol. 35, 2022. [https://arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339).

## Multimodality (LMM)
id: multimodalitaet
en: Multimodality / LMM
tags: fundamentals, architecture
level: basic

Refers to the ability of AI models (Large Multimodal Models) to process different data types—such as text, images, audio, and video—not only in isolation but to understand them in a common semantic space. Unlike earlier systems that combined different models (e.g., an image recognition AI passing text to an LLM), LMMs are "natively" multimodally trained. They can, for example, analyze a meme by simultaneously capturing the visual content and cultural meaning of the text and understanding their ironic interplay.

## RLHF vs. RLAIF
id: rlaif
en: RLAIF
tags: training
level: advanced

Reinforcement Learning from Human Feedback (RLHF) uses human evaluations to reward or punish the model. **Reinforcement Learning from AI Feedback (RLAIF)** automates this process by having a strong AI model evaluate the outputs of another model. RLAIF is crucial for scaling since human feedback is expensive and slow.

* **RLHF:** Christiano, Paul F., et al. "Deep Reinforcement Learning from Human Preferences". _Advances in Neural Information Processing Systems_, Vol. 30, 2017. [https://arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741).
* **RLAIF:** Lee, Harrison, et al. "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback". _arXiv preprint_, 2023. [https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267)

## Prompt Injection
id: prompt-injection
en: Prompt Injection
tags: safety
level: intermediate

An attack vector where malicious natural language inputs override an LLM's original instructions and cause the model to exhibit unintended behavior. The attack exploits the architectural inability of LLMs to distinguish between trusted developer instructions and untrusted user inputs. A distinction is made between **direct prompt injection** (malicious input by the user) and **indirect prompt injection** (hidden instructions in external data sources such as websites or documents).

* Perez, Fábio & Ribeiro, Ian. "Ignore Previous Prompt: Attack Techniques for Language Models". NeurIPS ML Safety Workshop 2022. [https://arxiv.org/abs/2211.09527](https://arxiv.org/abs/2211.09527).
* Greshake, Kai, et al. "Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection". Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, 2023, 79–90. [https://arxiv.org/abs/2302.12173](https://arxiv.org/abs/2302.12173).
* Video: [Prompt Injection, Jailbreaking, and Red Teaming – Sander Schulhoff](https://youtu.be/J9982NLmTXg)

## Jailbreaking
id: jailbreak
en: Jailbreaking
tags: safety
level: intermediate

An adversarial attack technique where specially constructed prompts bypass an LLM's safety mechanisms to force outputs that would normally be blocked or censored. Unlike [[#Prompt Injection]], jailbreaking explicitly targets circumventing trained-in safety guardrails (safety training/[[#Alignment]]), rather than overriding system instructions. Wei et al. identify two failure modes: **competing objectives** (conflict between capabilities and safety goals) and **mismatched generalization** (safety training fails in domains where the model has capabilities).

* Wei, Alexander, Nika Haghtalab & Jacob Steinhardt. "Jailbroken: How Does LLM Safety Training Fail?" Advances in Neural Information Processing Systems 36 (NeurIPS 2023). [https://arxiv.org/abs/2307.02483](https://arxiv.org/abs/2307.02483).
* Video: [Prompt Injection, Jailbreaking, and Red Teaming – Sander Schulhoff](https://youtu.be/J9982NLmTXg)

## Model Collapse
id: model-collapse
en: Model Collapse
tags: training, safety
level: advanced

Model Collapse refers to an irreversible degenerative learning process that occurs when generative models are trained primarily on the output of predecessor models—i.e., on [[#Synthetic Data]]—over multiple generations. Since statistical models tend to amplify probable patterns and smooth out rare edge events ("tails"), this recursive feedback loop causes the variance of the original distribution to be gradually lost. The model "forgets" the true data distribution, initially leading to the disappearance of rare nuances and ultimately resulting in a heavily simplified, repetitive representation of reality with minimal diversity. This underscores the necessity of always curating synthetic training data with original, human-made data.

* Shumailov, Ilia, et al. "The Curse of Recursion: Training on Generated Data Makes Models Forget". _Nature_, 2024. [https://arxiv.org/abs/2305.17493](https://arxiv.org/abs/2305.17493).

## Test-Time Compute
id: test-time-compute
en: Test-Time Compute
tags: architecture, prompting
level: advanced

A paradigm where compute is not only massively invested in model training but is deliberately deployed during [[#Inference]] ("test time"). Instead of immediately predicting the next token like classical LLMs (System-1 thinking: intuitive, fast), the model uses additional compute time to internally simulate different solution paths, correct errors, and verify steps (System-2 thinking: analytical, slow) before outputting an answer. This approach is the core of "Reasoning Models" like OpenAI o1.

* OpenAI o1 System Card. https://openai.com/index/learning-to-reason-with-llms/

## Test-Time Adaptation
id: test-time-adaptation
en: Test-Time Adaptation
tags: architecture, training
level: advanced

The process by which an AI model does not simply retrieve its frozen training knowledge but expends additional compute during [[#Inference]] (test time) to specifically adapt to the problem at hand.

* Test-Time Adaptation: A New Frontier in AI. https://youtu.be/C6sSs6NgANo

# Resources

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary
