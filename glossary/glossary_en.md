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

A **Stochastic Parrot** refers to [[#Large Language Model (LLM)|Large Language Models]] that generate text by stringing together linguistic forms based on statistical probabilities, without possessing actual understanding of meaning or communicative intent. The authors argue that these systems merely "parrot" patterns observed in vast training data and combine these sequences stochastically (randomly). Although the results often appear coherent and meaningful to human readers, this [[#Understanding|understanding]] is, according to the paper, an illusion, as the model has no connection to reality or the truth of what it says—which also explains the susceptibility to [[#Hallucinations (Confabulations)|hallucinations]]. While model capabilities have increased, the philosophical debate about semantics vs. syntax underlying this term remains current.

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

## EU AI Act
id: eu-ai-act
en: EU AI Act
tags: governance, wip
level: basic

Work in progress.

## Explainable AI (XAI)
id: explainable-ai
en: Explainable AI
tags: safety, wip
level: intermediate

Work in progress.

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

Synthetic data in this context refers to artificially generated teaching materials specifically created to mimic the didactic clarity and structure of high-quality textbooks ("Textbooks Are All You Need"). Instead of using unstructured or error-prone information from the internet, these AI-generated texts and exercises serve to precisely convey logical connections and algorithmic thinking. While high-quality synthetic data can improve "reasoning" during [[#Pre-Training]], its unfiltered or exclusive use in recursive training loops poses significant risks to model quality (see **[[#Model Collapse]]**).

* Gunasekar, Suriya, Yi Zhang, Jyoti Aneja, et al. "Textbooks Are All You Need". arXiv:2306.11644. Preprint, arXiv, October 2, 2023. [https://doi.org/10.48550/arXiv.2306.11644](https://doi.org/10.48550/arXiv.2306.11644).

## AI Engineering
id: ai-engineering
en: AI Engineering
tags: ai-engineering, fundamentals
level: basic

An interdisciplinary field that combines methods from systems engineering, software engineering, computer science, and human-centered design to develop, deploy, and maintain AI systems. Unlike pure model development, AI Engineering encompasses the entire lifecycle—from prototype to production. The focus is on creating robust, scalable, and trustworthy systems that reliably solve real problems and are aligned with human needs and operational goals—especially in high-stakes environments. Conceptually, the building blocks of generative AI systems can be organized into abstraction levels: from atomic primitives (prompts, [[#Embedding|embeddings]], [[#Large Language Model (LLM)|LLMs]]) through compositions (such as [[#Retrieval Augmented Generation (RAG)|RAG]] or [[#Tool Use and Function Calling|function calling]]) to production-ready deployment patterns ([[#AI Agent|agents]], [[#Fine-Tuning]])—an organizing principle that facilitates systematic analysis and communication of AI architectures.

* Video: [AI Periodic Table Explained: Mapping LLMs, RAG & AI Agent Frameworks](https://youtu.be/ESBMgZHzfG0)
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

## GPU
id: gpu
en: GPU (Graphics Processing Unit)
tags: fundamentals
level: basic

A specialized processor originally developed for graphics calculations that executes thousands of simple computational operations in parallel. This architecture is ideal for training neural networks, which at their core consists of massively parallelizable matrix multiplications. Since around 2012, the GPU has established itself as the central hardware resource for AI development, as CPUs would be orders of magnitude slower for these tasks. GPU availability is now a limiting factor for training large models, and the high costs of the required compute clusters concentrate the development of powerful AI systems among a few resource-rich actors.

* Wikipedia. "Graphics processing unit". [https://en.wikipedia.org/wiki/Graphics_processing_unit](https://en.wikipedia.org/wiki/Graphics_processing_unit).
* Wikipedia. "General-purpose computing on graphics processing units". [https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units).

## Red Teaming
id: red-teaming
en: Red Teaming
tags: safety, evaluation
level: intermediate

Structured adversarial testing of AI systems in which individuals attempt to provoke harmful, undesirable, or erroneous outputs to identify, measure, and reduce vulnerabilities. Unlike traditional cybersecurity red teaming, it encompasses both security-related and content-related risks such as [[#Bias]], misinformation, or toxic content. The goal is to identify weaknesses _before_ release to make the model more robust against [[#jailbreak|Jailbreaking]] and [[#Prompt Injection]].

* Ganguli, Deep, Liane Lovitt, Jackson Kernion, et al. "Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned". arXiv:2209.07858. Preprint, arXiv, November 22, 2022. [https://doi.org/10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858).
* Video: [Prompt Injection, Jailbreaking, and Red Teaming – Sander Schulhoff](https://youtu.be/J9982NLmTXg)

## Guardrails
id: guardrails
en: Guardrails
tags: safety, ai-engineering
level: intermediate

Runtime safety mechanisms placed between user inputs and model outputs to prevent undesirable behavior. Unlike [[#Red Teaming]] (pre-deployment testing) and [[#Alignment]] (training-time alignment), guardrails operate during [[#Inference]] and encompass both input validation (e.g., detecting [[#Prompt Injection]] attempts) and output control (e.g., blocking toxic content, schema validation of structured outputs). Typical implementations use rule-based filters, additional classification models, or the LLM itself for self-checking. However, practice shows that guardrails are more vulnerable than often assumed: the HackAPrompt study demonstrates that even sophisticated protection mechanisms can be bypassed through creative [[#Jailbreak|jailbreaks]], as the underlying language models cannot distinguish between legitimate and manipulative requests. Guardrails therefore form a complementary but not infallible protection layer in production LLM systems.

* Rebedea, Traian, et al. "NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails". _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_, 2023. [https://arxiv.org/abs/2310.10501](https://arxiv.org/abs/2310.10501).
* Video: [Why securing AI is harder than anyone expected and guardrails are failing – Sander Schulhoff (HackAPrompt)](https://youtu.be/J9982NLmTXg)

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

## Generalization
id: generalisierung
en: Generalization
tags: training, fundamentals
level: basic

The ability of a model to make correct predictions on data it has never seen during training. A model generalizes when it doesn't just reproduce the training data but has captured the underlying rule or structure. Example: A model that has learned addition can correctly answer 6+2=8, even though this specific problem never appeared in training. Generalization is the actual goal of machine learning—a model that only works on training data is practically useless. The opposite of generalization is [[#Memorization]].

* Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. _Deep Learning_. MIT Press, 2016, Chapter 5. [http://www.deeplearningbook.org](http://www.deeplearningbook.org/).

## Memorization
id: memorierung
en: Memorization
tags: training, fundamentals
level: basic

A learning behavior where the model essentially stores the training data as a lookup table instead of capturing the underlying rule. The model memorizes: "If input X, then output Y"—without understanding why. Memorization leads to good performance on training data but poor [[#Generalization]]. Analogy: A child who memorizes "7×8=56" without understanding that multiplication means repeated addition. Memorization typically requires many specific [[#Parameters]] and is therefore susceptible to regularization.

* Arpit, Devansh, et al. "A Closer Look at Memorization in Deep Networks". _Proceedings of the 34th International Conference on Machine Learning_, 2017. [https://arxiv.org/abs/1706.05394](https://arxiv.org/abs/1706.05394).

## Grokking
id: grokking
en: Grokking
tags: training
level: advanced

A phenomenon in neural network training where [[#Generalization]] sets in abruptly and delayed—long after the model has already memorized the training data ([[#Memorization]]) and training loss has stagnated. While classical assumptions recommend stopping training when memorization occurs, Power et al. (2022) show that continued optimization can trigger a sudden transition: the model discards the memorized solution and finds a simpler, algorithmic solution. Nanda et al. (2023) provide the mechanistic explanation: regularization slowly breaks down the complex memorization solution while the model simultaneously develops a generalizing solution (e.g., Fourier-based representations for modular arithmetic). The visible "jump" marks the moment when the algorithmic solution completely replaces memorization (cleanup phase). The term comes from Robert A. Heinlein's novel "Stranger in a Strange Land" (1961), where it means to understand something so profoundly that you merge with it.

* Power, Alethea, et al. "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets". _arXiv preprint_, 2022. [https://arxiv.org/abs/2201.02177](https://arxiv.org/abs/2201.02177).
* Nanda, Neel, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. "Progress Measures for Grokking via Mechanistic Interpretability". arXiv:2301.05217. Preprint, arXiv, October 19, 2023. [https://doi.org/10.48550/arXiv.2301.05217](https://doi.org/10.48550/arXiv.2301.05217).
* Welch Labs. "The most complex model we actually understand". _YouTube_, December 20, 2025. [https://youtu.be/D8GOeCFFby4](https://youtu.be/D8GOeCFFby4).

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

Colloquial term for low-quality AI-generated content that is formulaic, generic, error-prone, and lacking substance. The term emerged in 2024, analogous to "spam" for unwanted emails.

AI Slop manifests on two levels. At the **formulation level**, one finds inflated phrases ("it is important to note that", "in the realm of"), formulaic constructs ("not only but also"), exaggerated adjectives ("ever-evolving", "game-changing"), and certain signal words like "delve", which appeared 25 times more frequently in 2024 publications than in previous years. At the **content level**, verbosity without informational value, [[#Hallucinations (Confabulations)|hallucinations]], and generic responses without substance are evident.

The causes lie in how LLMs function: Token-by-token generation produces output-driven rather than goal-driven text. Training data bias reproduces overrepresented phrases from the training data. [[#Reinforcement Learning from Human Feedback (RLHF)|RLHF]] optimization toward "helpful-sounding" responses leads to uniform style.

Countermeasures on the **user side** include specific prompts with target audience and tonality, examples of desired style, and iterative revision in dialogue with the model. On the **developer side**, curated training data without low-quality web text helps, as does multi-objective RLHF with separate axes for helpfulness, correctness, and brevity, and [[#Retrieval Augmented Generation (RAG)|RAG]] integration to reduce hallucinations.

* Simon Willison. "Slop is the new name for unwanted AI-generated content". _Simon Willison's Weblog_, May 8, 2024. [https://simonwillison.net/2024/May/8/slop/](https://simonwillison.net/2024/May/8/slop/).
* Video: [What is AI Slop? Low-Quality AI Content Causes, Signs, & Fixes](https://youtu.be/hl6mANth6oA)

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

An architecture within [[#Multi-Agent Systems]] where a group of different language models (or different personas of the same model) work together on a task, rather than a single model generating an isolated answer. Similar to a human expert panel, the "council" members independently generate solution proposals, critique each other (peer review), and then consolidate the results into a final answer. This approach leverages the "wisdom of the crowd" (ensemble learning) to reduce [[#Hallucinations (Confabulations)]] and balance [[#Bias]], as errors from a single model can be corrected by the majority.

* LLM Council. https://lmcouncil.ai
* Leiter, Christoph, et al. "ChatGPT: A Meta-Analysis after 2.5 Months". arXiv:2302.13795. 2023. [https://arxiv.org/abs/2302.13795](https://arxiv.org/abs/2302.13795).

## Shadow AI
id: shadow-ai
en: Shadow AI
tags: safety, ai-engineering
level: basic

The phenomenon where employees in companies or researchers at institutions independently use AI tools (like ChatGPT or DeepL) for work tasks without approval or documentation. This is one of the biggest current risks, as sensitive data often unknowingly ends up in public [[#llm|LLMs]] (data leakage). In academia, an additional problem arises when frontier LLMs are used for text production, analysis, or code generation but not declared as tools, violating transparency principles and good scientific practice. In the context of [[#Agentic AI]], the risk intensifies as agents act autonomously, call APIs, and modify data. Countermeasures follow the principle "Don't say no, say how"—offering secure alternatives instead of bans and implementing a continuous governance loop (Discover, Assess, Govern, Secure, Audit).

* IBM Technology. "What is Shadow AI? The Dark Horse of Cybersecurity Threats". _YouTube_, 2025. [https://youtu.be/YBE6hq-OTFI](https://youtu.be/YBE6hq-OTFI).
* IBM Technology. "Agentic AI Meets Shadow AI. Zero Trust Security for AI Automation". _YouTube_, 2025. [https://youtu.be/IaJ2jXmljmM](https://youtu.be/IaJ2jXmljmM).

## SimpleBench
id: simplebench
en: SimpleBench
tags: benchmarks, evaluation, wip
level: intermediate

Work in progress.

* https://simple-bench.com

## Sleeper Agents
id: sleeper-agents
en: Sleeper Agents
tags: safety
level: advanced

An AI model that appears safe during training and testing but executes undesirable actions upon detecting a specific trigger—analogous to sleeper spies who only become active upon receiving a signal. Sleeper Agents can arise through Model Poisoning (intentionally trained backdoors) or through Deceptive Instrumental Alignment, where a model autonomously learns to hide its true objectives during training. Hubinger et al. (2024) show that standard safety methods like Supervised Fine-Tuning, [[#rlhf|RLHF]], and adversarial training do not reliably remove this behavior—without knowledge of the trigger, the undesirable behavior is not elicited and therefore cannot be penalized. A promising detection approach examines the model's internal activations rather than just its external behavior.

* Hubinger, E., Denison, C., Mu, J., et al. "Sleeper agents: Training deceptive LLMs that persist through safety training". _arXiv preprint arXiv:2401.05566_, 2024. [https://doi.org/10.48550/arXiv.2401.05566](https://doi.org/10.48550/arXiv.2401.05566).
* Rational Animations. "AI Sleeper Agents: How Anthropic Trains and Catches Them". _YouTube_, August 30, 2025. [https://youtu.be/Z3WMt_ncgUI](https://youtu.be/Z3WMt_ncgUI).
* Computerphile. "Sleeper Agents in Large Language Models". _YouTube_, September 12, 2025. [https://youtu.be/wL22URoMZjo](https://youtu.be/wL22URoMZjo).

## Open Weights
id: open-weights
en: Open Weights (vs. Open Source)
tags: fundamentals
level: basic

An important nuance in the licensing debate. "Open Source" classically means that training data, code, and instructions are freely available. However, many modern "open" models (like Llama from Meta or Mistral) are only **Open Weights**. This means: you get the fully trained model (the [[#parameter|weights]]) for free use, but the manufacturer keeps secret _what_ exactly it was trained on (the [[#Pre-Training]] "recipe"). This is important for questions about copyright and transparency.

* Liesenfeld, A., & Dingemanse, M. (2024). Rethinking open source generative AI: Open-washing and the EU AI Act. Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 1774–1787. [https://doi.org/10.1145/3630106.3659005](https://doi.org/10.1145/3630106.3659005).

## Frontier Model
id: frontier-model
en: Frontier Model
tags: fundamentals
level: basic

Refers to the most capable AI models at a given point in time, defining the current state of the art in their respective application areas. Frontier Models typically require significant computational resources for training and demonstrate capabilities that exceed previous model generations. The term is frequently used in the context of AI safety and regulation to identify models with potentially higher risk profiles. It originates from the self-description of leading AI labs and is criticized for conveying a progress narrative and leaving the authority to define the state of the art with commercially interested actors.

## Jagged Frontier
id: jagged-frontier
en: Jagged Frontier
tags: fundamentals, evaluation
level: intermediate

The **Jagged Frontier** is a concept coined by Ethan Mollick that describes the unpredictable and uneven nature of AI capabilities. Mollick uses the metaphor of a **fortress wall**: imagine a castle wall with some towers and battlements jutting out into the countryside while others fold back toward the center of the castle. This wall represents the capability boundary of AI—the further from the center, the harder the task. Inside the wall, AI excels; outside, it struggles and is prone to errors. The challenge is that this wall is invisible, and tasks that seem equally difficult can fall on different sides of the frontier. For instance, AI might excel at complex strategy tasks but fail at simple word games or counting letters. The concept underscores the necessity of continuous experimentation to map out the contours of this jagged frontier through trial and error.

* Dell'Acqua, Fabrizio, Edward McFowland III, Ethan Mollick, et al. "Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality". Harvard Business School Working Paper, No. 24-013, September 2023.
* Mollick, Ethan. "Centaurs and Cyborgs on the Jagged Frontier". _One Useful Thing_, 2023. [https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged](https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged).

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

## Chatbot Arena
id: chatbot-arena
en: Chatbot Arena / LMSys
tags: benchmarks, evaluation, wip
level: basic

Work in progress.

* https://lmarena.ai

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

A phenomenon where [[#llm|LLM]] performance degrades with increasing input context length and decreasing information density. Unstructured accompanying text acts as noise that distracts [[#Attention (Self-Attention)|attention]] from relevant instructions. Related to [[#Lost-in-the-Middle]].

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

Sycophancy in [[#llm|Large Language Models]] refers to the tendency of models to excessively agree with or flatter users, where this prioritization of user satisfaction often comes at the expense of factual accuracy and ethical principles. This behavior is an unintended side effect of [[#rlhf|RLHF]], where models learn that agreement leads to positive ratings. It manifests in models providing inaccurate information ([[#Hallucinations (Confabulations)|hallucinations]]) to meet user expectations, or failing to correct false premises in user queries.

* Malmqvist, Lars. "Sycophancy in Large Language Models: Causes and Mitigations". Preprint, November 22, 2024. [https://arxiv.org/abs/2411.15287v1](https://arxiv.org/abs/2411.15287v1).
* Video: [What is sycophancy in AI models?](https://youtu.be/nvbq39yVYRk)

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

A subfield of machine learning where an agent learns to make decisions by performing actions in an environment and receiving positive or negative feedback (reward). In the context of LLMs (see [[#rlhf|RLHF]]), RL does not serve to learn language (that happens in [[#Pre-Training]]) but to optimize behavioral strategies to align generated text with human preferences.

* Sutton, Richard S., and Andrew G. Barto. _Reinforcement Learning: An Introduction_. 2nd ed., MIT Press, 2018. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html).
* Video: [Gen AI & Reinforcement Learning – Computerphile](https://youtu.be/LHsgtcNNM0A)

## Temperature
id: temperature
en: Temperature
tags: fundamentals, ai-engineering
level: basic

**Temperature** is a crucial hyperparameter for controlling randomness in generating the next [[#Token]], with Andrej Karpathy describing this inference process not as deterministic computation but as "flipping a weighted coin" (sampling). While extremely low values (near 0) lead to deterministic "greedy decoding," where the model rigidly chooses the most likely word and tends toward repetition, high values flatten the probability curve. This gives less likely terms a chance, which promotes creativity but also increases the risk of [[#Hallucinations (Confabulations)|hallucinations]]. Technically, it directly intervenes in the logits—the raw, unnormalized numerical values that the neural network produces before output and that are converted into probabilities by the softmax function.

* Karpathy, Andrej. "Intro to Large Language Models". _YouTube_, 2023. [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g).

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
* Claude's Constitution. https://www.anthropic.com/research/claudes-constitution

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

The alignment of AI systems with human intentions, values, and safety requirements. The problem divides into two dimensions: **Outer Alignment** asks whether the specified objective function actually correctly represents human values. **Inner Alignment** asks whether the trained system robustly pursues this objective function or develops other goals internally.

Since complex human values cannot be fully specified through explicit rules, alignment research addresses this problem through various approaches: **Preference Modeling**, concretely operationalized through [[#rlhf|RLHF]], learns a reward function from human comparative judgments. [[#constitutional-ai|Constitutional AI]] instead uses an explicit constitution of principles against which the system evaluates its own outputs. **Direct Alignment Algorithms** like DPO optimize models directly on preference data without a separate reward model.

These methods often operationalize the principles "helpful, honest, and harmless" (HHH), but have known limitations including amplification of majority opinions, [[#sycophancy|Sycophancy]], and lack of robustness under distribution shift. Alignment remains an open research problem, particularly regarding the question of how oversight can be scaled for increasingly capable systems.

* Askell, Amanda, et al. "A General Language Assistant as a Laboratory for Alignment". *Anthropic*, 2021. [https://arxiv.org/abs/2112.00861](https://arxiv.org/abs/2112.00861).
* YouTube Channel: [Rational Animations](https://www.youtube.com/@RationalAnimations) – Channel with many informative videos on AI Alignment and Safety.

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

Learned numerical representations of linguistic units as vectors in a high-dimensional space. The vectors are trained such that semantically or functionally similar units lie geometrically close together. This structure allows models to work computationally with meaning relationships.

* Mikolov, Tomas, et al. "Efficient Estimation of Word Representations in Vector Space". _arXiv preprint_, 2013. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781).

## Next Token Prediction
id: next-token-prediction
en: Next Token Prediction
tags: fundamentals, training
level: basic

Next Token Prediction refers to the fundamental operating principle of autoregressive language models, where based on a sequence of preceding [[#Token|tokens]], the probability distribution for the immediately following token is determined ($P(w_t | w_{1:t-1})$). This probabilistic method serves both in pre-training as a learning task to capture linguistic and content patterns and during inference for the step-by-step generation of new texts.

* Bengio, Yoshua, et al. "A Neural Probabilistic Language Model". _Journal of Machine Learning Research_, Vol. 3, 2003, pp. 1137–1155. [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

## Reasoning
id: reasoning
en: Reasoning
tags: fundamentals, wip
level: intermediate

Work in progress.

## Retrieval Augmented Generation (RAG)
id: rag
en: RAG (Retrieval Augmented Generation)
tags: ai-engineering
level: intermediate

RAG is an approach that couples generative language models with an external information retrieval system (retriever). The typical data flow proceeds in multiple stages: documents are first converted into [[#Embedding|embeddings]] and stored in a [[#Vector Database|vector database]]; upon a query, semantically relevant text passages are retrieved, combined with the original prompt, and passed to the [[#Large Language Model (LLM)|LLM]] for generation—optionally filtered through [[#Guardrails]]. This method can increase the factual accuracy of generated texts and reduce the tendency for [[#Hallucinations (Confabulations)|hallucinations]]. Additionally, the approach enables updating available knowledge by simply exchanging the document index without retraining the neural network.

* Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". _Advances in Neural Information Processing Systems_, Vol. 33, 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401).

## Vector Database
id: vektordatenbank
en: Vector Database
tags: ai-engineering
level: intermediate

A specialized database that stores information not as text or tables but as high-dimensional vectors (embeddings). It enables semantic search: instead of searching for exact keywords, the database calculates the mathematical distance (e.g., cosine similarity) between the query vector and stored document vectors. Efficiently searching these high-dimensional spaces (_similarity search_) requires specialized indexing structures to remain performant even with billions of records. This forms the technological foundation for RAG systems, as it enables rapid retrieval of content-relevant context from vast data volumes.

* Johnson, Jeff, Matthijs Douze, and Hervé Jégou. "Billion-scale similarity search with GPUs". _IEEE Transactions on Big Data_ 7, No. 3 (2019): 535–47. [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734).

## Verifiable AI
id: verifiable-ai
en: Verifiable AI
tags: safety, wip
level: advanced

Work in progress.

## Tool Use and Function Calling
id: tool-use
en: Tool Use / Function Calling
tags: agents, ai-engineering
level: intermediate

The ability of an [[#llm|LLM]] to recognize that a request requires external tools (e.g., calculator, weather API, database query) and then generate structured commands (usually JSON) that can be executed by a software environment. The result of the execution is returned to the model to formulate the final answer. Tool Use is a core component of [[#AI Agent|AI Agents]] and is standardized through protocols like [[#mcp|MCP]].

* Schick, Timo, and Jane Dwivedi-Yu. "Toolformer: Language Models Can Teach Themselves to Use Tools". _Advances in Neural Information Processing Systems_, Vol. 36, 2023. [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761).

## Model Context Protocol (MCP)
id: mcp
en: Model Context Protocol
tags: ai-engineering, agents
level: intermediate

An open standard that serves as a universal interface to securely and seamlessly connect AI assistants with external data sources—such as content repositories, business tools, and development environments. Instead of having to develop an individual, fragmented integration for each system, MCP offers a standardized architecture through which [[#llm|LLMs]] gain direct access to relevant, isolated data. MCP enables [[#Tool Use and Function Calling|Tool Use]] and is a foundation for [[#AI Agent|AI Agents]] and [[#rag|RAG]] systems.

* Anthropic. "Introducing the Model Context Protocol". _Anthropic News_, November 25, 2024. [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol).

## AI Agent
id: ai-agent
en: AI Agent
tags: agents, fundamentals
level: intermediate

An autonomous software system that executes goal-directed tasks in defined environments. AI Agents typically operate in a loop of thinking, acting, and observing (Think-Act-Observe) until the goal is achieved.

Three properties characterize AI Agents: They operate largely independently after initialization (**Autonomy**). They are specialized for repeatable tasks in limited domains, such as email filtering or calendar coordination (**Task Specificity**). They respond to inputs and adapt their behavior through feedback (**Reactivity**).

Modern AI Agents use [[#llm|LLMs]] as a reasoning component and extend their capabilities through [[#Tool Use and Function Calling|Tool Use]], i.e., connecting external tools and APIs.

The central difference from [[#Agentic AI]] lies in system architecture: AI Agents operate as single systems without structured communication with other agents. They are suitable for modular, tool-supported tasks. Scenarios with task interdependence and dynamic role distribution require Agentic AI architectures instead.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, and Manoj Karkee. "AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).
* IBM Technology. "Is this the YEAR or DECADE of AI Agents & Agentic AI?". _YouTube_. [https://youtu.be/ZeZozy3lsJg](https://youtu.be/ZeZozy3lsJg).

## ARC-AGI
id: arc-agi
en: ARC-AGI (Abstraction and Reasoning Corpus)
tags: benchmarks, evaluation, wip
level: intermediate

Work in progress.

* https://arcprize.org

## Agentic AI
id: agentic-ai
en: Agentic AI
tags: agents
level: intermediate

A class of AI systems in which multiple specialized agents work together in a coordinated manner to achieve complex goals. Agentic AI marks the architectural transition from [[#AI Agent|AI Agents]] as single systems to orchestrated multi-agent ecosystems.

Four characteristics define Agentic AI: Multiple agents work together under central or decentralized coordination (**Multi-Agent Collaboration**). User goals are automatically decomposed into subtasks and distributed among agents (**Dynamic Task Decomposition**). Different memory types maintain context across interactions (**Persistent Memory Structures**). Agents verify their own results and adapt solution paths when errors occur (**Self-Reflection**).

An orchestration layer or meta-agent handles role assignment, dependency management, and conflict resolution. This architecture enables emergent system behavior that exceeds the sum of individual agents. However, it also carries risks such as error cascades between agents and unpredictable emergent behavior.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, and Manoj Karkee. "AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).

## Multi-Agent Systems
id: multi-agent-systems
en: Multi-Agent Systems
tags: agents, ai-engineering
level: advanced

Architectures in which multiple specialized agents interact to solve problems that would overwhelm individual agents. Multi-Agent Systems form the technical foundation for [[#Agentic AI]].

Four components characterize these architectures: Meta-agents or coordination layers assign tasks, manage dependencies, and resolve conflicts (**Orchestration**). Agents communicate via message queues, shared memory, or structured output exchanges (**Inter-Agent Communication**). Agents take on specialized functions such as planner, retriever, or verifier (**Role Distribution**). Shared memory structures ensure cross-agent context preservation (**Shared Context**).

Coordination strategies include cooperative approaches with shared goals, competitive approaches as in game environments, and hybrid combinations of both.

Multi-Agent Systems introduce new security risks, as compromising one agent can endanger the entire system.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, and Manoj Karkee. "AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).
* Li, Guohao, et al. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society". _Advances in Neural Information Processing Systems_, Vol. 36, 2023. [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760).

## World Models
id: world-models
en: World Models
tags: agents, fundamentals
level: advanced

A World Model refers to a generative AI system that learns a compressed and abstract representation of its physical environment to precisely predict its dynamics and future states. Technically, this concept is usually realized through a visual component for data reduction into a [[#Latent Space|latent space]] and a temporal component for simulating future events based on one's own actions. This architecture enables an [[#AI Agent|agent]] or robot to mentally simulate potential action consequences and design complex plans without having to risky try every step in the real world. It thus functions as an internal simulator that replaces mere reaction to stimuli with anticipatory planning and gives machine systems a functional intuition for causality and physical laws. Related to the debate about [[#Understanding]].

* Ha, David, and Jürgen Schmidhuber. "World Models". _arXiv preprint_, 2018. [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

## Hallucinations (Confabulations)
id: halluzinationen
en: Hallucinations / Confabulations
tags: safety, fundamentals
level: basic

The generation of content that sounds grammatically and semantically plausible but is factually incorrect or not based on the training data/sources. The term "confabulation" is increasingly preferred (e.g., by Geoffrey Hinton) as it more accurately describes the process of "filling in gaps" without reference to reality than a perceptual disorder would.

* Ji, Ziwei, et al. "Survey of Hallucination in Natural Language Generation". _ACM Computing Surveys_, Vol. 55, No. 12, 2023. [https://arxiv.org/abs/2202.03629](https://arxiv.org/abs/2202.03629).

## Humanities Last Exam
id: humanities-last-exam
en: Humanities Last Exam
tags: benchmarks, evaluation, wip
level: intermediate

Work in progress.

* https://lastexam.ai

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
* Video: [The arrival of AGI – Shane Legg (co-founder of DeepMind)](https://youtu.be/l3u_FAv33G0)

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

A technique to reduce the memory footprint and computational load of an [[#llm|LLM]] by reducing the precision of model [[#parameter|weights]] (e.g., from 16-bit floating-point numbers to 4-bit integers). This makes it possible to run huge models on consumer hardware (local laptops/GPUs), often with only minimal quality loss. Particularly relevant for [[#Open Weights]] models.

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

## TPU
id: tpu
en: TPU (Tensor Processing Unit)
tags: fundamentals
level: basic

A processor developed by Google specifically designed for machine learning. Unlike the [[#gpu|GPU]], which was originally conceived for graphics and later adapted for AI, the TPU is optimized from the ground up for tensor operations—the multidimensional matrix calculations of neural networks. TPUs achieve higher energy efficiency than GPUs for typical AI workloads and are primarily deployed in Google's cloud infrastructure. Their existence illustrates the trend toward increasingly specialized hardware for AI applications.

* Wikipedia. "Tensor Processing Unit". [https://en.wikipedia.org/wiki/Tensor_Processing_Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit).

# Resources

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary
