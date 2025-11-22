# Glossar zu Generativer KI und Large Language Models

## Large Language Model (LLM)
id: llm
tags: grundlagen, architektur
level: basic

Ein Large Language Model ist ein künstliches neuronales Netz, das auf riesigen Textmengen trainiert wurde, um menschenähnliche Sprache zu verstehen und zu generieren. LLMs wie GPT, Claude oder Gemini nutzen Transformer-Architektur und zeichnen sich durch Milliarden von Parametern aus, die statistische Muster über Sprache kodieren. Diese Modelle können durch Prompting eine Vielzahl von Sprachaufgaben erfüllen, ohne aufgabenspezifisch trainiert zu werden.

* Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762
* Brown et al. (2020). Language Models are Few-Shot Learners. https://arxiv.org/abs/2005.14165

## Context Window
id: context-window
tags: grundlagen, architektur
level: basic

Das Context Window definiert die maximale Textmenge (gemessen in Token), die ein LLM in einer einzelnen Interaktion verarbeiten kann. Es umfasst sowohl den Input-Prompt als auch die generierte Antwort. Moderne LLMs haben Context Windows von 8.000 Token (ca. 6.000 Wörter) bis über 1 Million Token. Ein größeres Context Window ermöglicht es dem Modell, Kohärenz über längere Gespräche zu bewahren und größere Dokumente zu verarbeiten.

* Anthropic (2024). Introducing Claude 3.5 Sonnet with 200K context window. https://www.anthropic.com/claude

## Prompt Engineering
id: prompt-engineering
tags: grundlagen, prompting
level: basic

Prompt Engineering ist die Praxis, Texteingaben (Prompts) so zu gestalten und zu verfeinern, dass LLMs effektiv zu gewünschten Ergebnissen geführt werden. Es umfasst das Verstehen, wie Modelle Anweisungen interpretieren, das Bereitstellen klaren Kontexts, die Verwendung von Beispielen (Few-Shot Learning) und das iterative Verbessern von Prompts basierend auf Modellantworten. Effektives Prompt Engineering kann die Ausgabequalität erheblich verbessern, ohne das Modell selbst zu verändern.

* Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903
* OpenAI (2023). Prompt Engineering Guide. https://platform.openai.com/docs/guides/prompt-engineering

## RAG (Retrieval-Augmented Generation)
id: rag
tags: ai-engineering, techniken
level: intermediate

RAG ist eine Technik, die LLM-Ausgaben durch Kombination mit externer Wissensabfrage verbessert. Wenn eine Anfrage eingeht, werden zunächst relevante Dokumente oder Textpassagen aus einer Wissensbasis abgerufen (oft mittels Vektorähnlichkeitssuche), dann dem LLM als Kontext bereitgestellt. Dieser Ansatz reduziert Halluzinationen, ermöglicht Modellen den Zugriff auf aktuelle Informationen und verankert Antworten in konkreten Quellmaterialien. RAG ist besonders wertvoll für domänenspezifische Anwendungen.

* Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401

## Token
id: token
tags: grundlagen, architektur
level: basic

Ein Token ist die grundlegende Texteinheit, die LLMs verarbeiten. Tokenisierung zerlegt Text in kleinere Teile – typischerweise Wörter, Teilwörter oder Zeichen. Zum Beispiel könnte "Verständnis" in "Ver", "ständ" und "nis" als drei Token aufgeteilt werden. Die Anzahl der Token in einem Text beeinflusst Verarbeitungskosten und Context Window-Nutzung. Im Durchschnitt entspricht ein Token etwa 0,75 Wörtern im Englischen (im Deutschen etwas weniger).

* OpenAI (2023). What are tokens and how to count them? https://help.openai.com/en/articles/4936856

## Fine-tuning
id: fine-tuning
tags: training, techniken
level: intermediate

Fine-tuning ist der Prozess, bei dem ein vortrainiertes LLM auf einem kleineren, aufgabenspezifischen Datensatz weitertrainiert wird, um es für bestimmte Anwendungsfälle oder Domänen anzupassen. Dieser Prozess justiert die Modellparameter zur Optimierung der Leistung für die Zielaufgabe, während allgemeine Sprachfähigkeiten erhalten bleiben. Fine-tuning benötigt weniger Daten und Rechenressourcen als Training von Grund auf, aber mehr als reines Prompt Engineering.

* Howard & Ruder (2018). Universal Language Model Fine-tuning for Text Classification. https://arxiv.org/abs/1801.06146

## Halluzination
id: hallucination
tags: grundlagen, sicherheit, limitationen
level: basic

Halluzination bezeichnet das Phänomen, wenn ein LLM Informationen generiert, die plausibel erscheinen, aber faktisch inkorrekt oder erfunden sind. Dies geschieht, weil LLMs darauf trainiert sind, statistisch wahrscheinliche Textmuster vorherzusagen, nicht aber faktische Richtigkeit zu verifizieren. Halluzinationen sind eine fundamentale Limitation aktueller LLM-Architekturen und können teilweise durch Techniken wie RAG, Fact-Checking und sorgfältiges Prompt-Design gemildert werden.

* Zhang et al. (2023). Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. https://arxiv.org/abs/2309.01219

## Few-Shot Learning
id: few-shot-learning
tags: prompting, techniken
level: intermediate

Few-Shot Learning ist eine Technik, bei der eine kleine Anzahl von Beispielen (typischerweise 2-10) im Prompt bereitgestellt wird, um die gewünschte Aufgabe oder das Ausgabeformat zu demonstrieren. Das LLM lernt aus diesen Beispielen innerhalb des Context Windows und wendet das Muster auf neue Eingaben an. Dieser Ansatz liegt zwischen Zero-Shot (keine Beispiele) und Fine-Tuning und bietet eine flexible Möglichkeit, Modellverhalten zu steuern, ohne zusätzliches Training.

* Brown et al. (2020). Language Models are Few-Shot Learners. https://arxiv.org/abs/2005.14165

## Chain-of-Thought (CoT)
id: chain-of-thought
tags: prompting, techniken, reasoning
level: intermediate

Chain-of-Thought Prompting ist eine Technik, die LLMs dazu ermutigt, komplexe Denkaufgaben in Zwischenschritte zu zerlegen, bevor sie zu einer finalen Antwort gelangen. Indem das Modell explizit aufgefordert wird, "Schritt für Schritt zu denken" oder Beispiele mit Denkspuren gezeigt werden, verbessert CoT die Leistung bei Aufgaben, die logisches Denken, Mathematik oder mehrstufige Problemlösung erfordern.

* Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. https://arxiv.org/abs/2201.11903
* Kojima et al. (2022). Large Language Models are Zero-Shot Reasoners. https://arxiv.org/abs/2205.11916

## Temperature
id: temperature
tags: grundlagen, parameter
level: intermediate

Temperature ist ein Parameter, der die Zufälligkeit der Ausgabe eines LLM steuert. Niedrigere Temperaturen (z.B. 0,0-0,3) machen Ausgaben deterministischer und fokussieren auf die wahrscheinlichsten Token, während höhere Temperaturen (z.B. 0,7-1,0) Diversität und Kreativität erhöhen, aber Kohärenz verringern können. Temperature ist entscheidend, um je nach Anwendungsfall zwischen faktischer Genauigkeit und kreativer Generierung zu balancieren.

* OpenAI (2023). API Reference: Temperature parameter. https://platform.openai.com/docs/api-reference/chat/create#temperature

## Embeddings
id: embeddings
tags: architektur, techniken
level: intermediate

Embeddings sind numerische Vektorrepräsentationen von Text, die semantische Bedeutung in einem hochdimensionalen Raum erfassen. Wörter, Sätze oder Dokumente mit ähnlichen Bedeutungen haben ähnliche Embedding-Vektoren. LLMs verwenden Embeddings intern, und sie werden auch für Aufgaben wie semantische Suche, Clustering und RAG eingesetzt. Moderne Embedding-Modelle können ganze Absätze oder Dokumente als einzelne Vektoren darstellen.

* Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space. https://arxiv.org/abs/1301.3781

## System Prompt
id: system-prompt
tags: prompting, techniken
level: basic

Ein System Prompt ist eine spezielle Anweisung, die einem LLM zu Beginn einer Konversation gegeben wird und seine Rolle, sein Verhalten und seine Beschränkungen definiert. Anders als User-Prompts bleiben System Prompts typischerweise über die gesamte Konversation bestehen und setzen den Gesamtkontext dafür, wie das Modell antworten soll. Sie werden häufig verwendet, um Expertise-Domänen, Ausgabeformate oder Verhaltensrichtlinien zu spezifizieren.

* Anthropic (2024). System Prompts Documentation. https://docs.anthropic.com/claude/docs/system-prompts

## GraphRAG
id: graphrag
tags: ai-engineering, techniken, neue-paradigmen-2024-25, wip
level: research

GraphRAG erweitert traditionelles RAG, indem abgerufene Informationen in Wissensgraphen statt flachen Dokumentensammlungen organisiert werden. Dieser Ansatz erfasst Beziehungen zwischen Entitäten und Konzepten und ermöglicht anspruchsvolleres Reasoning über Verbindungen in den Daten. GraphRAG ist besonders nützlich für komplexe Domänen wie historische Forschung, wissenschaftliche Literatur oder juristische Dokumente, wo Beziehungen zwischen Entitäten entscheidend sind.

* Microsoft Research (2024). GraphRAG: Unlocking LLM discovery on narrative private data. https://www.microsoft.com/en-us/research/blog/graphrag/

## Transformer-Architektur
id: transformer
tags: architektur, grundlagen
level: research

Der Transformer ist die neuronale Netzwerkarchitektur, die modernen LLMs zugrunde liegt, eingeführt im Paper "Attention Is All You Need". Er nutzt Self-Attention-Mechanismen, um Eingabesequenzen parallel statt sequenziell zu verarbeiten, was effizientes Training auf großen Datensätzen ermöglicht. Kernkomponenten umfassen Multi-Head Attention, Positional Encodings und Feed-Forward-Layer. Die Transformer-Architektur revolutionierte NLP und bildet die Basis von Modellen wie GPT, BERT und Claude.

* Vaswani et al. (2017). Attention Is All You Need. https://arxiv.org/abs/1706.03762

## RLHF (Reinforcement Learning from Human Feedback)
id: rlhf
tags: training, alignment, sicherheit
level: research

RLHF ist eine Trainingstechnik zur Ausrichtung von LLMs an menschliche Präferenzen und Werte. Nach dem initialen Pre-Training werden Modelle mittels Feedback von menschlichen Evaluatoren, die verschiedene Antworten bewerten, feinabgestimmt. Dieser Prozess hilft Modellen, hilfreicher, harmloser und ehrlicher zu werden. RLHF ist entscheidend für die Entwicklung sicherer, benutzerfreundlicher KI-Assistenten, wirft aber auch Fragen auf, wessen Werte kodiert werden.

* Ouyang et al. (2022). Training language models to follow instructions with human feedback. https://arxiv.org/abs/2203.02155
* Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback. https://arxiv.org/abs/2212.08073
