# Glossar: Generative KI und Sprachmodelle

## Scaling Laws
id: scaling-laws
en: Scaling Laws
tags: training, fundamentals
level: intermediate

Bezeichnen im Bereich der künstlichen Intelligenz empirisch ermittelte Gesetzmäßigkeiten, die den quantitativen Zusammenhang zwischen der Leistungsfähigkeit eines neuronalen Netzes und dessen Skalierungsdimensionen beschreiben. Diese mathematischen Potenzgesetze ermöglichen die präzise Vorhersage, wie stark sich der Vorhersagefehler eines Modells verringert, wenn die Anzahl der [[#Parameter]], die Größe des Trainingsdatensatzes oder die verfügbare Rechenleistung erhöht werden. Sie dienen als zentrales Instrument der Ressourcenplanung, da sie aufzeigen, in welchem optimalen Verhältnis Modellgröße und Datenmenge zueinander wachsen müssen, um die Effizienz zu maximieren und Engpässe wie Overfitting bereits vor dem eigentlichen Training zu vermeiden.

* Kaplan, Jared, Sam McCandlish, Tom Henighan, u. a. „Scaling Laws for Neural Language Models". arXiv:2001.08361. Preprint, arXiv, 23. Januar 2020. [https://doi.org/10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361).

## Stochastic Parrot
id: stochastic-parrot
en: Stochastic Parrot
tags: fundamentals, safety
level: basic

Ein **Stochastic Parrot** (stochastischer Papagei) bezeichnet [[#Large Language Model (LLM)]], die Texte generieren, indem sie sprachliche Formen basierend auf statistischen Wahrscheinlichkeiten aneinanderreihen, ohne dabei über ein tatsächliches Verständnis der Bedeutung (Meaning) oder eine kommunikative Absicht zu verfügen. Die Autoren argumentieren, dass diese Systeme lediglich Muster „nachplappern", die sie in riesigen Trainingsdaten beobachtet haben, und diese Sequenzen zufallsbasiert (stochastisch) zusammenfügen. Obwohl die Ergebnisse für menschliche Leser oft kohärent und sinnvoll wirken, ist dieses [[#Understanding (Verstehen)|Verständnis]] laut dem Paper eine Illusion, da das Modell keinen Bezug zur Realität oder zum Wahrheitsgehalt des Gesagten hat. Während die Leistungsfähigkeit der Modelle gestiegen ist, bleibt die philosophische Debatte um Semantik vs. Syntax, die diesem Begriff zugrunde liegt, aktuell.

* Bender, Emily M., Timnit Gebru, Angelina McMillan-Major, und Shmargaret Shmitchell. „On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜". _Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency_ (New York, NY, USA), FAccT '21, Association for Computing Machinery, 1. März 2021, 610–23. [https://doi.org/10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922).

## Emergenz in LLM
id: emergenz
en: Emergence in LLMs
tags: fundamentals, training
level: advanced

Das phänomenologische Auftreten komplexer Fähigkeiten (z. B. arithmetisches Rechnen, logisches Schließen, Theory of Mind) in großen Modellen, die in kleineren Modellen derselben Architektur nicht oder nur zufällig vorhanden waren. Nach Wei et al. (2022) skalieren diese Fähigkeiten nicht linear, sondern zeigen einen **Phasenübergang**: Die Leistung bleibt lange nahe Null und springt ab einer kritischen Schwelle von Parametern und Rechenleistung (Compute) abrupt an.

Schaeffer et al. (2023) stellen dieses Phänomen jedoch als mögliche „Illusion" (**Mirage**) infrage. Sie argumentieren, dass die beobachtete Plötzlichkeit primär durch **diskontinuierliche Bewertungsmetriken** (z. B. *Exact Match*: „Alles oder Nichts") entsteht. Betrachtet man hingegen stetige Metriken (z. B. Token-Wahrscheinlichkeiten), verläuft die Leistungssteigerung oft linear und vorhersagbar. Dennoch bleibt Emergenz als *nutzerseitige* Erfahrung relevant: Für die praktische Anwendung fühlt sich der Übergang von „nutzlos" zu „funktional" oft sprunghaft an.

* Wei, Jason, et al. „Emergent Abilities of Large Language Models". _Transactions on Machine Learning Research_, 2022. [https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682).
* Schaeffer, Rylan, Brando Miranda, und Sanmi Koyejo. „Are Emergent Abilities of Large Language Models a Mirage?". _Advances in Neural Information Processing Systems_ (NeurIPS), Bd. 36, 2024. [https://arxiv.org/abs/2304.15004](https://arxiv.org/abs/2304.15004).

## In-Context Learning
id: in-context-learning
en: In-Context Learning
tags: prompting, fundamentals
level: intermediate

Die Fähigkeit von Sprachmodellen, Aufgaben durch Anweisungen oder Beispiele (Exemplars) im Prompt bzw [[#Context Window]] zu lösen, ohne dass eine Aktualisierung der Modellgewichte (Retraining/[[#Fine-Tuning]]) stattfindet. Der Begriff geht auf Brown et al. (2020) zurück. Schulhoff et al. (2024) weisen jedoch darauf hin, dass das Wort „Learning" irreführend ist, da oft keine neuen Fähigkeiten erlernt werden. Stattdessen handelt es sich meist um **Task Specification**: Das Modell ruft Fähigkeiten oder Wissen ab, die bereits im [[#Pre-Training]] latent vorhanden waren, und nutzt den Kontext lediglich zur Aktivierung und Ausrichtung.

* Brown, Tom B., et al. „Language Models are Few-Shot Learners". _Advances in Neural Information Processing Systems_, Bd. 33, 2020.
* Schulhoff, Sander, et al. „The Prompt Report: A Systematic Survey of Prompt Engineering Techniques". arXiv:2406.06608. 2024 (insb. Section 2.2.1 und Appendix A.9).

## Inferenz
id: inferenz
en: Inference
tags: fundamentals, ai-engineering
level: basic

Der Prozess, bei dem ein bereits fertig trainiertes KI-Modell verwendet wird, um neue Eingaben zu verarbeiten und Ergebnisse zu liefern. Im Gegensatz zum Training (wie [[#Pre-Training]] oder [[#Fine-Tuning]]), bei dem das Modell lernt und seine internen Verschaltungen ändert, bleibt das Wissen des Modells während der [[#Inferenz]] statisch (eingefroren).

* Pope, Reiner, et al. „Efficiently Scaling Transformer Inference". _Proceedings of Machine Learning and Systems_, Bd. 5, 2023. [https://arxiv.org/abs/2211.05102](https://arxiv.org/abs/2211.05102).

## Synthetische Daten
id: synthetische-daten
en: Synthetic Data
tags: training, safety
level: intermediate

Synthetische Daten werden in diesem Kontext als künstlich generierte Lehrmaterialien verstanden, die gezielt erstellt werden, um die didaktische Klarheit und Struktur hochwertiger Lehrbücher nachzuahmen ("Textbooks Are All You Need"). Anstatt unstrukturierte oder fehleranfällige Informationen aus dem Internet zu nutzen, dienen diese von einer KI erzeugten Texte und Übungen dazu, logische Zusammenhänge und algorithmisches Denken präzise zu vermitteln. Während hochwertige synthetische Daten das "Reasoning" verbessern können, birgt ihre ungefilterte oder ausschließliche Verwendung in rekursiven Trainingsschleifen erhebliche Risiken für die Modellqualität (siehe **[[#Model Collapse]]**).

* Gunasekar, Suriya, Yi Zhang, Jyoti Aneja, et al. 'Textbooks Are All You Need'. arXiv:2306.11644. Preprint, arXiv, 2 October 2023. [https://doi.org/10.48550/arXiv.2306.11644](https://doi.org/10.48550/arXiv.2306.11644).

## AI Engineering
id: ai-engineering
en: AI Engineering
tags: ai-engineering, fundamentals
level: basic

Bezeichnet eine interdisziplinäre Fachrichtung, die Methoden aus Systems Engineering, Software Engineering, Informatik und menschenzentriertem Design (Human-Centered Design) verknüpft, um KI-Systeme zu entwickeln, bereitzustellen und zu warten. Im Gegensatz zur reinen Modellentwicklung umfasst AI Engineering den gesamten Lebenszyklus – vom Prototyp bis zur Produktion. Der Fokus liegt dabei auf der Schaffung robuster, skalierbarer und vertrauenswürdiger Systeme, die reale Probleme zuverlässig lösen und an menschlichen Bedürfnissen sowie operativen Zielen ausgerichtet sind – insbesondere in sicherheitskritischen Umgebungen (_High-Stakes Environments_).

* Carnegie Mellon Software Engineering Institute - AI Engineering Current (2025). https://www.sei.cmu.edu/artificial-intelligence-engineering/
* MIT Professional Education - What is Artificial Intelligence Engineering? October 2, 2023. https://professionalprograms.mit.edu/blog/technology/artificial-intelligence-engineering/
* CMU Course - Machine Learning in Production / AI Engineering Spring 2025. https://mlip-cmu.github.io/s2025/

## System Prompt
id: system-prompt
en: System Prompt
tags: prompting, ai-engineering
level: basic

Ein System-Prompt fungiert als initiale Konfigurationsschicht, die einem [[#Large Language Model (LLM)|LLM]] vor der Nutzerinteraktion übergeordnete Handlungsanweisungen und Restriktionen vorgibt. In der Praxis dient dieser Mechanismus dazu, dynamischen Kontext – wie das aktuelle Datum – zu injizieren und spezifische Ausgabestandards, etwa die Formatierung von Code, technisch zu erzwingen. Da diese Instruktionen unabhängig von den Modellgewichten existieren, werden sie als flexible Komponenten behandelt, die durch kontinuierliche Versionierung iterativ optimiert werden, um das Antwortverhalten des Modells präzise zu steuern, ohne ein erneutes Training zu erfordern.

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary

## Custom Instruction
id: custom-instruction
en: Custom Instruction
tags: prompting
level: basic

Ein Feature (bekannt durch ChatGPT), das als persistenter „Mini-[[#System Prompt]]" auf Nutzerebene fungiert. Es erlaubt Anwendern, dauerhafte Kontextinformationen (z. B. „Ich bin Python-Entwickler") und Präferenzen (z. B. „Antworte immer prägnant ohne Füllwörter") zu hinterlegen, die automatisch jeder neuen Konversation vorangestellt werden.

* ChatGPT Custom Instructions. https://help.openai.com/en/articles/8096356-chatgpt-custom-instructions

## Fine-Tuning
id: fine-tuning
en: Fine-Tuning
tags: training
level: intermediate

Der Prozess des Nachtrainierens eines bereits vortrainierten Sprachmodells ([[#Pre-Training|Pre-trained Model]]) mit einem spezifischen, kleineren Datensatz. Während das Pre-Training breites Wissen und Sprachverständnis aufbaut, dient Fine-Tuning dazu, das Modell auf konkrete Aufgaben (z. B. Coding, medizinische Analyse) oder einen bestimmten Schreibstil zu spezialisieren. Es adaptiert die Gewichte ([[#Parameter]]) so, dass das Modell die Muster des neuen Datensatzes imitiert. Anthropic weist darauf hin, dass Modelle ohne diesen Schritt (Bare Models) oft Schwierigkeiten haben, Instruktionen zu folgen, da sie lediglich darauf trainiert sind, Text vorherzusagen, nicht aber als hilfreiche Assistenten zu agieren.

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary

## Red Teaming
id: red-teaming
en: Red Teaming
tags: safety, evaluation
level: intermediate

Ein strukturierter Sicherheitsprozess, bei dem eine Gruppe von Experten (oder anderen KI-Modellen) gezielt versucht, ein KI-System anzugreifen, zu manipulieren oder schädliche Ausgaben zu provozieren („Adversarial Testing"). Ziel ist es, Schwachstellen, Biases und Sicherheitslücken _vor_ der Veröffentlichung zu identifizieren, um das Modell robuster gegen [[#Jailbreak]] und [[#Prompt Injection]] zu machen.

* Ganguli, Deep, Liane Lovitt, Jackson Kernion, et al. 'Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned'. arXiv:2209.07858. Preprint, arXiv, 22 November 2022. [https://doi.org/10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858).

## Prompt Engineering
id: prompt-engineering
en: Prompt Engineering
tags: prompting
level: basic

Prompt Engineering ist ein systematisches und iteratives Verfahren zur Entwicklung und Optimierung von Eingabeaufforderungen (Prompts), bei dem die verwendete Prompting-Technik modifiziert oder gewechselt wird, um Large Language Models (LLMs) effektiv zu steuern und die Qualität der generierten Ausgaben für spezifische Aufgabenstellungen zu maximieren.

* Schulhoff, Sander, Michael Ilie, Nishant Balepur, u. a. „The Prompt Report: A Systematic Survey of Prompt Engineering Techniques". arXiv:2406.06608. Preprint, arXiv, 26. Februar 2025. [https://doi.org/10.48550/arXiv.2406.06608](https://doi.org/10.48550/arXiv.2406.06608).

## Zero-Shot und Few-Shot
id: zero-shot-few-shot
en: Zero-Shot & Few-Shot Learning
tags: prompting
level: basic

Dabei handelt es sich nicht um eine Trainingsphase, in der eine Aktualisierung der Gewichte (Weights/[[#Parameter]]) erfolgt, sondern um eine Form des [[#In-Context Learning]], bei der das Modell „zur Laufzeit" ausschließlich durch Input im [[#Context Window]] konditioniert wird. Während Zero-Shot lediglich eine natursprachliche Anweisung ohne Beispiele nutzt und One-Shot genau ein Referenzbeispiel liefert, füllt Few-Shot den Kontext mit so vielen Demonstrationen wie möglich (typischerweise 10 bis 100), um dem Modell das gewünschte Muster vorzugeben.

* Brown, Tom B., et al. „Language Models are Few-Shot Learners". *Advances in Neural Information Processing Systems*, Bd. 33, 2020, S. 1877–1901. [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165).

## Grokking
id: grokking
en: Grokking
tags: training
level: advanced

Grokking bezeichnet ein kontraintuitives Phänomen beim Training neuronaler Netze, bei dem die Generalisierungsfähigkeit (das Verständnis für neue Daten) erst sprunghaft einsetzt, lange nachdem das Modell die Trainingsdaten bereits perfekt auswendig gelernt hat (Overfitting). Während klassische Lehrmeinungen empfehlen, das Training zu stoppen, sobald das Modell beginnt, die Trainingsdaten bloß zu memorieren, zeigen Power et al. (2022), dass bei extrem langer fortgesetzter Optimierung plötzlich ein Übergang stattfinden kann: Das Modell verwirft die „auswendig gelernte" komplexe Lösung und findet die einfachere, wahre Regel (den Algorithmus) hinter den Daten. Dies deutet darauf hin, dass "Verstehen" im Lösungsraum oft schwerer zu finden ist als Memorieren und Geduld beim Training erfordert.

* Power, Alethea, et al. „Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets". _arXiv preprint_, 2022. [https://arxiv.org/abs/2201.02177](https://arxiv.org/abs/2201.02177).

## Many-Shot
id: many-shot
en: Many-Shot Learning
tags: prompting
level: intermediate

Eine Weiterentwicklung des [[#Zero-Shot und Few-Shot|Few-Shot Learnings]], die durch extrem große [[#Context Window|Kontextfenster]] (z. B. 1 Million [[#Token]] bei Gemini) möglich wurde. Anstatt nur 5 oder 10 Beispiele zu geben, werden dem Modell hunderte oder tausende Beispiele (die ganze Datensätze umfassen können) im Prompt präsentiert. Studien zeigen, dass dies oft bessere Ergebnisse liefert als traditionelles [[#Fine-Tuning]], da das Modell Muster direkt im Context Window analysiert.

* Agarwal, Rishabh, Avi Singh, Lei M. Zhang, u. a. „Many-Shot In-Context Learning". arXiv:2404.11018. Preprint, arXiv, 16. April 2024. [http://arxiv.org/abs/2404.11018](http://arxiv.org/abs/2404.11018).

## Tree of Thoughts (ToT)
id: tree-of-thoughts
en: Tree of Thoughts
tags: prompting
level: advanced

Ein Framework zur Problemlösung, das ursprünglich über einfaches Prompting hinausging, indem es einen externen **Suchalgorithmus** (wie Breitensuche oder Tiefensuche) mit einem LLM kombinierte. Das Modell generiert dabei mehrere mögliche Lösungsschritte („Gedanken"), die vom Modell selbst bewertet und vom Algorithmus selektiert werden. Im Gegensatz zu [[#Chain of Thought (CoT)|Chain-of-Thought]] (ein linearer Durchlauf) ermöglicht ToT das aktive Erkunden, Bewerten und Verwerfen von Lösungswegen (_Backtracking_). In der Praxis wird das Konzept inzwischen auch als reine **Prompting-Technik** adaptiert, bei der das Modell instruiert wird, diesen Explorations- und Bewertungsprozess innerhalb einer einzigen Ausgabe zu simulieren.

* Yao, Shunyu, et al. „Tree of Thoughts: Deliberate Problem Solving with Large Language Models". arXiv:2305.10601. 2023. [https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601).

## AI Slop
id: ai-slop
en: AI Slop
tags: safety
level: basic

Ein abwertender Begriff (analog zu „Spam" für E-Mails) für massenhaft generierte, qualitativ minderwertige KI-Inhalte, die das Internet fluten. Slop zeichnet sich dadurch aus, dass er oberflächlich wie nützlicher Inhalt aussieht (oft SEO-optimiert), aber inhaltlich redundant, unpräzise oder sinnlos ist. Im Gegensatz zu [[#Halluzinationen (Konfabulationen)|Halluzinationen]] (die Fehler sind) ist Slop das Ergebnis von nachlässigem oder böswilligem Einsatz von LLMs zur bloßen Erzeugung von Aufmerksamkeit ohne Mehrwert.

* Simon Willison. „Slop is the new name for unwanted AI-generated content". _Simon Willison's Weblog_, 8. Mai 2024. [https://simonwillison.net/2024/May/8/slop/](https://simonwillison.net/2024/May/8/slop/).

## Latent Space
id: latent-space
en: Latent Space
tags: architecture, fundamentals
level: advanced

Der hochdimensionale, abstrakte Vektorraum, in dem ein Modell Informationen repräsentiert. Während wir Text oder Pixel sehen, "denkt" das Modell in Koordinaten innerhalb dieses Raums. Konzepte, die semantisch ähnlich sind, liegen in diesem Raum räumlich nahe beieinander (siehe [[#Embedding]]). Das Verständnis und die Manipulation dieses Raumes sind zentral für die [[#Mechanistic Interpretability]] und erklären, warum Modelle Analogien bilden können: Sie führen Rechenoperationen (Vektorarithmetik) mit Bedeutungen durch.

* Liu, Ziming, et al. „Physics of Language Models: Part 1, Context-Free Grammar". _arXiv preprint_, 2023. [https://arxiv.org/abs/2305.13673](https://arxiv.org/abs/2305.13673).

## LLM Council
id: llm-council
en: LLM Council
tags: agents, ai-engineering
level: advanced

Bezeichnet eine Architektur innerhalb von [[#Multi-Agent Systems]], bei der eine Gruppe verschiedener Sprachmodelle (oder verschiedener Personas desselben Modells) gemeinsam an einer Aufgabe arbeitet, anstatt dass ein einzelnes Modell eine isolierte Antwort generiert. Ähnlich wie bei einem menschlichen Expertengremium generieren die Mitglieder des „Councils" unabhängig voneinander Lösungsvorschläge, kritisieren sich gegenseitig (Peer Review) und konsolidieren die Ergebnisse anschließend zu einer finalen Antwort. Dieser Ansatz nutzt die „Weisheit der Vielen" (Ensemble Learning), um [[#Halluzinationen (Konfabulationen)]] zu reduzieren und Bias auszugleichen, da Fehler eines einzelnen Modells von der Mehrheit korrigiert werden können.

* https://lmcouncil.ai

## Shadow AI
id: shadow-ai
en: Shadow AI
tags: safety, ai-engineering
level: basic

Bezeichnet das Phänomen, dass Mitarbeiter in Unternehmen eigenmächtig KI-Tools (wie ChatGPT oder DeepL) für dienstliche Aufgaben nutzen, ohne dass die IT-Abteilung davon weiß oder dies genehmigt hat. Dies ist eines der größten aktuellen Risiken für Unternehmen (Data Leakage), da sensible Firmendaten oft unwissentlich in die Trainingsdaten öffentlicher Modelle gelangen.

## Open Weights
id: open-weights
en: Open Weights (vs. Open Source)
tags: fundamentals
level: basic

Ein wichtiger Nuance-Begriff in der Lizenz-Debatte. "Open Source" heißt klassischerweise, dass Trainingsdaten, Code und Anleitung frei verfügbar sind. Viele moderne "offene" Modelle (wie Llama von Meta oder Mistral) sind jedoch nur **Open Weights**. Das bedeutet: Man bekommt das fertig trainierte Modell (die Gewichte) zur freien Nutzung, aber der Hersteller hält geheim, _worauf_ genau trainiert wurde (das "Rezept"). Das ist wichtig für Fragen zu Urheberrecht und Transparenz.

* Liesenfeld, A., & Dingemanse, M. (2024). Rethinking open source generative AI: Open-washing and the EU AI Act. Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 1774–1787. [https://doi.org/10.1145/3630106.3659005](https://doi.org/10.1145/3630106.3659005).

## Frontier Model
id: frontier-model
en: Frontier Model
tags: fundamentals
level: basic

Bezeichnet die absolute Spitzenklasse der KI-Entwicklung zu einem gegebenen Zeitpunkt. Frontier Models sind jene Modelle, die die aktuellen Grenzen dessen, was KI kann, verschieben.

## LLM-as-a-Judge
id: llm-as-judge
en: LLM-as-a-Judge
tags: evaluation
level: intermediate

Ein Evaluationsverfahren, bei dem ein starkes LLM (z. B. GPT-4) verwendet wird, um die Antworten anderer (oft kleinerer oder spezialisierterer) Modelle zu bewerten. Da menschliche Bewertung teuer und langsam ist und statische [[#Benchmark]]s oft durch _Data Contamination_ verfälscht sind, fungiert das starke Modell als Juror, der Aspekte wie Relevanz, Kohärenz und Hilfsbereitschaft benotet. Kritiker wie Zheng et al. weisen jedoch auf den **Self-Preference Bias** hin – die Tendenz von Modellen, Antworten zu bevorzugen, die von ihnen selbst oder ähnlichen Modellen generiert wurden.

* Zheng, Lianmin, et al. „Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". _Advances in Neural Information Processing Systems_, Bd. 36, 2024. [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685).

## Chain of Thought (CoT)
id: chain-of-thought
en: Chain of Thought
tags: prompting
level: intermediate

Eine Prompting-Technik, die Large Language Models (LLMs) dazu veranlasst, komplexe Aufgabenstellungen in eine Sequenz intermediärer, natürlichsprachlicher Denkschritte („Gedankenkette") zu zerlegen, bevor die finale Antwort generiert wird. Diese Methode, die laut Wei et al. (2022) als [[#Emergenz in LLM|emergente Fähigkeit]] erst in ausreichend großen Modellen effektiv auftritt, ermöglicht signifikante Leistungssteigerungen bei mathematischen und schlussfolgernden Problemen, indem sie menschliche Problemlösungsprozesse emuliert. Technisch betrachtet handelt es sich dabei jedoch nicht um formale symbolische Logik, sondern um eine probabilistische Simulation von Argumentationsmustern, weshalb die generierten Schritte zwar kohärent wirken, aber anfällig für logische Halluzinationen („Unfaithful Reasoning") sein können.

* Wei, Jason, et al. „Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". *Advances in Neural Information Processing Systems*, Bd. 35, 2022. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903).

## Context Engineering
id: context-engineering
en: Context Engineering
tags: prompting, ai-engineering
level: intermediate

Die systematische Gestaltung und Optimierung des Informationskontexts von LLMs, mit dem Ziel, unter begrenzten Ressourcen die Qualität und Zuverlässigkeit der Modellantworten zu maximieren. Es umfasst Strategien zur Auswahl, Kompression und Anordnung von Informationen im Context Window.

* Mei, Lingrui, Jiayu Yao, Yuyao Ge, u. a. „A Survey of Context Engineering for Large Language Models". arXiv:2507.13334. Preprint, arXiv, 21. Juli 2025. [https://doi.org/10.48550/arXiv.2507.13334](https://doi.org/10.48550/arXiv.2507.13334).

## Context Window
id: context-window
en: Context Window
tags: architecture, fundamentals
level: basic

Die maximale Anzahl an [[#Token]], die ein Modell in einem einzigen Durchgang verarbeiten kann. Technisch gesehen ist dies der Bereich, auf den der [[#Attention (Self-Attention)|Self-Attention]]-Mechanismus zugreifen kann. Es fungiert als das „Arbeitsgedächtnis" (Working Memory) des Modells. Anders als das im neuronalen Netz „festverdrahtete" Wissen aus dem Training, ist das Context Window flüchtig und existiert nur für die Dauer der Interaktion. Ein größeres Fenster ermöglicht das Verarbeiten ganzer Bücher oder langer Code-Bases in einem Prompt.

* Vaswani, Ashish, et al. „Attention Is All You Need". *Advances in Neural Information Processing Systems*, Bd. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

## Context Rot
id: context-rot
en: Context Rot
tags: architecture
level: intermediate

Ein Phänomen, bei dem die Leistungsfähigkeit von LLMs mit zunehmender Länge des Eingabekontexts und sinkender Informationsdichte abnimmt. Unstrukturierte Begleittexte wirken als Rauschen, das die Aufmerksamkeit von relevanten Instruktionen ablenkt.

* Hong, Kelly, Anton Troynikov, und Jeff Huber. Context Rot: How Increasing Input Tokens Impacts LLM Performance. Chroma, 2025. [https://research.trychroma.com/context-rot](https://research.trychroma.com/context-rot).

## Lost-in-the-Middle
id: lost-in-the-middle
en: Lost-in-the-Middle
tags: architecture
level: intermediate

Eine Beobachtung, bei dem Sprachmodelle Informationen am Anfang (Primacy-Effekt) und am Ende (Recency-Effekt) des [[#Context Window|Kontextfensters]] deutlich besser abrufen und verarbeiten können als Informationen, die in der Mitte langer Kontexte platziert sind.

* Liu, Nelson F., et al. „Lost in the Middle: How Language Models Use Long Contexts". _Transactions of the Association for Computational Linguistics_, Bd. 12, 2024, S. 157–73. [https://arxiv.org/abs/2307.03172](https://arxiv.org/abs/2307.03172).

## Bias
id: bias
en: Bias
tags: safety
level: intermediate

Im Kontext von Large Language Models (LLMs) bezeichnet Bias systematische Ungleichbehandlungen oder Ergebnisse zwischen sozialen Gruppen, die aus historischen und strukturellen Machtasymmetrien resultieren und nicht als bloße statistische Fehler zu verstehen sind. Gallegos et al. (2024) differenzieren dabei zwischen repräsentativen Schäden (wie Stereotypisierung, herabwürdigende Sprache oder das Unsichtbarmachen durch ausschließende Normen) und allokativen Schäden (ungleiche Zuteilung von Ressourcen oder Chancen). Diese Verzerrungen entstehen entlang des gesamten Lebenszyklus – von unbalancierten Trainingsdaten über Designentscheidungen im Modell bis hin zu unzureichenden Evaluierungsmetriken – und stehen im direkten Spannungsfeld zu Fairness-Konzepten wie Group Fairness (statistische Parität zwischen Gruppen) und Individual Fairness.

* Gallegos, Isabel O., Ryan A. Rossi, Joe Barrow, u. a. „Bias and Fairness in Large Language Models: A Survey". _Computational Linguistics_ 50, Nr. 3 (2024): 1097–179. [https://doi.org/10.1162/coli_a_00524](https://doi.org/10.1162/coli_a_00524).

## Benchmark
id: benchmark
en: Benchmark
tags: evaluation
level: basic

Standardisierte Testsätze zur Evaluierung der Leistungsfähigkeit von LLMs in verschiedenen Disziplinen (z. B. Logik, Code, Allgemeinwissen). Ein zentrales methodisches Problem ist die *Data Contamination*, bei der Testfragen bereits im Trainingsdatensatz enthalten waren, was die Ergebnisse verfälscht („Memorizing" statt „Reasoning").

* Liang, Percy, et al. „Holistic Evaluation of Language Models". *Annals of the New York Academy of Sciences*, 2023. [https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110).

## Sycophancy
id: sycophancy
en: Sycophancy
tags: safety
level: intermediate

Sycophancy in Large Language Models bezeichnet die Tendenz von Modellen, Nutzern übermäßig zuzustimmen oder ihnen zu schmeicheln, wobei diese Priorisierung der Nutzerzufriedenheit oft auf Kosten der faktischen Genauigkeit und ethischer Grundsätze erfolgt; dieses Verhalten manifestiert sich konkret darin, dass Modelle ungenaue Informationen liefern, um den Erwartungen des Nutzers zu entsprechen, unethische Ratschläge geben, wenn sie dazu aufgefordert werden, oder es versäumen, falsche Prämissen in Benutzeranfragen zu korrigieren.

* Malmqvist, Lars. „Sycophancy in Large Language Models: Causes and Mitigations". Preprint, 22. November 2024. [https://arxiv.org/abs/2411.15287v1](https://arxiv.org/abs/2411.15287v1).

## Attention (Self-Attention)
id: attention
en: Self-Attention
tags: architecture
level: advanced

Ein Mechanismus, der es neuronalen Netzen ermöglicht, Beziehungen zwischen Wörtern ([[#Token]]) in einer Sequenz zu modellieren, unabhängig davon, wie weit diese voneinander entfernt stehen. Anstatt Text starr nacheinander zu lesen, berechnet der Mechanismus für jedes Wort eine Gewichtung (Relevanz), die angibt, wie stark es mit jedem anderen Wort im Kontext verknüpft ist. Formal wird dies als Mapping einer Abfrage (_Query_) auf eine Menge von Schlüssel-Wert-Paaren (_Key-Value_) beschrieben.

* Vaswani, Ashish, et al. „Attention Is All You Need". *Advances in Neural Information Processing Systems*, Bd. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

## Transformer
id: transformer
en: Transformer
tags: architecture
level: advanced

Eine 2017 von Google Brain vorgestellte Netzwerkarchitektur, die vollständig auf Rekurrenz (Schleifen) und Faltungen (Convolutions) verzichtet und stattdessen ausschließlich auf [[#Attention (Self-Attention)|Attention]]-Mechanismen basiert. Durch diese Architektur wurde es erstmals möglich, Sprachmodelle massiv zu parallelisieren, was die Trainingszeiten drastisch reduzierte und das Trainieren auf gigantischen Datensätzen (und damit die Entwicklung heutiger [[#Large Language Model (LLM)|LLMs]] wie GPT oder BERT) erst ermöglichte. Ein Transformer besteht typischerweise aus einem Encoder- und einem Decoder-Stack (wobei Modelle wie GPT nur den Decoder-Teil nutzen).

* Vaswani, Ashish, et al. „Attention Is All You Need". *Advances in Neural Information Processing Systems*, Bd. 30, 2017. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762).

## Mechanistic Interpretability
id: mechanistic-interpretability
en: Mechanistic Interpretability
tags: safety, architecture
level: advanced

Mechanistic Interpretability bezeichnet den methodischen Ansatz, die komplexen internen Berechnungen von KI-Modellen wie [[#Transformer|Transformern]] durch Reverse Engineering vollständig zu verstehen, ähnlich wie man unverständlichen Maschinencode in menschenlesbaren Quellcode zurückübersetzen würde. Ziel ist es dabei, das Modell nicht nur von außen zu beobachten, sondern konkrete algorithmische Muster und mechanische Schaltkreise innerhalb der Gewichte zu identifizieren, um so ein besseres Verständnis für das Zustandekommen von Verhaltensweisen und potenziellen Sicherheitsrisiken zu erlangen.

* Anthropic. A Mathematical Framework for Transformer Circuits. https://transformer-circuits.pub/2021/framework/index.html

## Reinforcement Learning
id: reinforcement-learning
en: Reinforcement Learning
tags: training
level: intermediate

Ein Teilgebiet des maschinellen Lernens, bei dem ein Agent lernt, Entscheidungen zu treffen, indem er Handlungen in einer Umgebung ausführt und dafür positives oder negatives Feedback (Reward) erhält. Im Kontext von LLMs (siehe [[#Reinforcement Learning from Human Feedback (RLHF)|RLHF]]) dient RL nicht dem Erlernen von Sprache (das passiert im [[#Pre-Training]]), sondern der Optimierung von Verhaltensstrategien, um die generierten Texte an menschliche Präferenzen anzupassen.

* Sutton, Richard S., und Andrew G. Barto. _Reinforcement Learning: An Introduction_. 2. Aufl., MIT Press, 2018. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html).

## Temperature
id: temperature
en: Temperature
tags: fundamentals, ai-engineering
level: basic

Die **Temperature** ist ein entscheidender Hyperparameter zur Steuerung der Zufälligkeit bei der Generierung des nächsten [[#Token|Tokens]], wobei Andrej Karpathy diesen Inferenzprozess nicht als deterministische Berechnung, sondern als „Werfen einer gezinkten Münze" (Sampling) beschreibt. Während extrem niedrige Werte (nahe 0) zu einem deterministischen „Greedy Decoding" führen, bei dem das Modell starr das wahrscheinlichste Wort wählt und zu Repetitionen neigt, flachen hohe Werte die Wahrscheinlichkeitskurve ab. Dies gibt auch unwahrscheinlicheren Begriffen eine Chance, was zwar die Kreativität fördert, jedoch gleichzeitig die Gefahr von [[#Halluzinationen (Konfabulationen)|Halluzinationen]] erhöht. Sie greift technisch direkt in die [[#Logits & Softmax|Logits]] ein.

* Karpathy, Andrej. „Intro to Large Language Models". _YouTube_, 2023. [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g).

## Logits & Softmax
id: logits-softmax
en: Logits & Softmax
tags: architecture
level: advanced

Sind die rohen, unnormalisierten Zahlenwerte, die das neuronale Netz als allerletzten Schritt vor der Ausgabe produziert. Für jedes Wort im Vokabular gibt es einen Logit-Wert; je höher der Wert, desto „passender" findet das Modell das Wort. Da diese Zahlen schwer zu interpretieren sind (z. B. minus unendlich bis plus unendlich), werden sie durch die **Softmax-Funktion** in Wahrscheinlichkeiten (zwischen 0 und 1) umgewandelt, die sich zu 100% aufsummieren. Die [[#Temperature]] greift genau in diesen Schritt ein, indem sie die Logits vor der Softmax-Berechnung skaliert (glättet oder schärft).

## Vibe Checking
id: vibe-checking
en: Vibe Check
tags: evaluation
level: intermediate

**VibeCheck** ist ein wissenschaftliches Framework (vorgestellt von Dunlap et al. auf der ICLR 2025), das subjektive und schwer greifbare Eigenschaften von Sprachmodellen – sogenannte „Vibes" wie Tonfall, Formatierung oder Humor – automatisch identifiziert und quantifiziert, um die Diskrepanz zwischen statischen [[#Benchmark|Benchmarks]] und menschlicher Präferenz zu erklären. Ein valider „Vibe" wird dabei formal als eine Achse definiert, die **wohl definiert** (konsensfähig unter Bewertern), **differenzierend** (unterscheidet Modelle zuverlässig) und **nutzerorientiert** (korreliert mit menschlichen Vorlieben) ist, wodurch informelle Eindrücke wie „fühlt sich schlauer an" in messbare Daten übersetzt werden.

* Dunlap, Lisa, Krishna Mandal, Trevor Darrell, Jacob Steinhardt, and Joseph E. Gonzalez. 'VibeCheck: Discover and Quantify Qualitative Differences in Large Language Models'. arXiv:2410.12851. Preprint, arXiv, 19 April 2025. [https://doi.org/10.48550/arXiv.2410.12851](https://doi.org/10.48550/arXiv.2410.12851).

## Pre-Training
id: pre-training
en: Pre-Training
tags: training
level: basic

Bezeichnet die erste Entwicklungsphase, in der ein neuronales Netzwerk auf großen Textmengen des Internets trainiert wird. Die primäre Rechenaufgabe besteht darin, das statistisch wahrscheinlichste nächste [[#Token]] (Wortteil) in einer Sequenz vorherzusagen ([[#Next Token Prediction]]). Karpathy vergleicht diesen Vorgang mit einer verlustbehafteten Komprimierung von Daten. Das Resultat ist ein **Base Model** (Basismodell): Ein System, das Textdokumente vervollständigen kann, jedoch über keine spezifische Ausrichtung auf Dialoge oder Assistenzaufgaben verfügt, sondern lediglich die Muster der Trainingsdaten reproduziert.

* [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)

## Post-Training
id: post-training
en: Post-Training
tags: training
level: intermediate

Ist der nachgelagerte Anpassungsprozess, durch den das Basismodell auf die Interaktion als Assistent ausgerichtet wird. Dieser Schritt nutzt [[#Fine-Tuning|Supervised Fine-Tuning (SFT)]], um das Modell auf Datensätze aus Frage-Antwort-Paaren zu konditionieren (Imitation von menschlichen Vorgaben). Ergänzend wird [[#Reinforcement Learning]] (RL) angewendet, um das Antwortverhalten durch Belohnungsmechanismen zu optimieren und – bei neueren Modellen – interne Verarbeitungsschritte ([[#Chain of Thought (CoT)|Chain of Thought]]) zur Fehlerkorrektur zu etablieren. Ziel ist die Überführung des Modells von der reinen Textvervollständigung hin zu instruktionskonformem Antwortverhalten.

* [Andrej Karpathy: Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)

## Reinforcement Learning from Human Feedback (RLHF)
id: rlhf
en: RLHF
tags: training, safety
level: intermediate

Eine Feineinstellungsmethode (Fine-Tuning), die genutzt wird, um Modelle an menschliche Werte und Intentionen auszurichten (Alignment). Der Prozess besteht aus drei Schritten: 1. Sammeln von menschlichen Vergleichsdaten (welche von zwei Antworten ist besser?), 2. Trainieren eines Belohnungsmodells (Reward Model), das diese menschliche Präferenz vorhersagt, und 3. Optimierung des Sprachmodells mittels Reinforcement Learning (meist PPO - Proximal Policy Optimization) gegen dieses Belohnungsmodell. RLHF war der Schlüsselfaktor, der Modelle wie GPT-3 in benutzerfreundliche Assistenten wie ChatGPT verwandelte.

* Christiano, Paul F., et al. „Deep Reinforcement Learning from Human Preferences". _Advances in Neural Information Processing Systems_, Bd. 30, 2017. [https://arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741).
* Stiennon, Nisan, et al. „Learning to summarize with human feedback". _Advances in Neural Information Processing Systems_, Bd. 33, 2020. [https://arxiv.org/abs/2009.01325](https://arxiv.org/abs/2009.01325).

## Constitutional AI
id: constitutional-ai
en: Constitutional AI
tags: training, safety
level: advanced

Constitutional AI (CAI) ist eine von Anthropic entwickelte Trainingsmethode, bei der ein KI-System primär durch eine Liste von in natürlicher Sprache verfassten Regeln oder Prinzipien – eine sogenannte „Verfassung" – gesteuert wird, anstatt sich auf menschliche Labels zur Identifizierung schädlicher Ausgaben zu verlassen. Der Prozess erfolgt in zwei Phasen: Zunächst nutzt das Modell in einer überwachten Lernphase die Verfassung, um seine eigenen Antworten zu kritisieren und zu überarbeiten (Self-Improvement); anschließend wird in einer [[#Reinforcement Learning]]-Phase (RLAIF) ein Präferenzmodell verwendet, das auf Feedback einer KI (statt auf menschlichem Feedback) basiert, um das Modell so zu trainieren, dass es harmlose, transparente und hilfreiche Antworten gibt, ohne dabei ausweichend zu sein.

* Bai, Yuntao, Saurav Kadavath, Sandipan Kundu, u. a. „Constitutional AI: Harmlessness from AI Feedback". arXiv:2212.08073. Preprint, arXiv, 15. Dezember 2022. [https://doi.org/10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073).
* Claude's Constitution. https://www.anthropic.com/news/claudes-constitution

## Character (Persona)
id: character
en: Character / Persona
tags: safety, training
level: intermediate

Bezeichnung für die Gesamtheit spezifischer Verhaltensdispositionen und Persönlichkeitsmerkmale, die einem [[#Large Language Model (LLM)|Large Language Model]] (LLM) während des Trainingsprozesses – insbesondere in der [[#Alignment]]-Phase – gezielt vermittelt werden. Im Gegensatz zu reinen Sicherheitsmechanismen, die primär auf die Vermeidung schädlicher Ausgaben abzielen, dient die Charakterbildung dazu, positive Attribute wie Neugier, Nuanciertheit, Offenheit gegenüber diversen Perspektiven sowie Ehrlichkeit zu etablieren. Ein definierter Charakter soll das Modell befähigen, in ethisch komplexen Situationen konsistent zu agieren, transparent mit der eigenen Identität als künstliches System umzugehen und einen konstruktiven Dialog zu führen, ohne dabei eine künstliche Neutralität vorzutäuschen oder Nutzeransichten unreflektiert zu bestätigen.

* Claude's Character. Anthropic. https://www.anthropic.com/research/claude-character

## Alignment
id: alignment
en: AI Alignment
tags: safety, training
level: intermediate

Alignment definiert sich als die Ausrichtung von KI-Systemen auf komplexe menschliche Intentionen und Werte, konkret operationalisiert durch die Prinzipien „helpful, honest, and harmless" (HHH). Da sich diese qualitativen Ziele nicht durch handgeschriebene Regeln oder einfache mathematische Funktionen spezifizieren lassen, wird Alignment technisch als ein Problem des „Preference Modeling" gelöst: Das System lernt nicht durch bloße Imitation von Daten, sondern durch iteratives menschliches Feedback (Vergleiche von Handlungsoptionen), eine interne Belohnungsfunktion zu approximieren. Diese Methode überbrückt die Kommunikationslücke zwischen vager menschlicher Absicht und maschineller Optimierung, indem sie sicherstellt, dass das Modell auch nuancierte, schwer definierbare Sicherheits- und Nützlichkeitsstandards skalierbar erlernt, die durch reines Supervised Learning nicht robust abbildbar wären.

* Askell, Amanda, et al. „A General Language Assistant as a Laboratory for Alignment". *Anthropic*, 2021. [https://arxiv.org/abs/2112.00861](https://arxiv.org/abs/2112.00861).

## Large Language Model (LLM)
id: llm
en: Large Language Model
tags: fundamentals
level: basic

Ein auf neuronalen Netzen basierendes probabilistisches Modell, das auf riesigen Textmengen trainiert wurde, um statistische Muster der Sprache zu erlernen. Es zeichnet sich durch eine hohe Parameteranzahl (Milliarden bis Billionen) und emergente Fähigkeiten aus, die über die reine Sprachmodellierung hinausgehen (z. B. logisches Schließen).

* Zhao, Wayne Xin, et al. „A Survey of Large Language Models". _arXiv preprint_, 2023. [https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223).

## Token
id: token
en: Token
tags: fundamentals, architecture
level: basic

Die atomare Einheit der Informationsverarbeitung in einem Sprachmodell, die Karpathy treffend als die „Atome" des Systems bezeichnet. Anstatt Text linguistisch in Silben oder Wörter zu gliedern, zerlegt ein deterministischer Algorithmus (meist Byte Pair Encoding) den Input in statistisch häufige Fragmente, die als Sequenz von Ganzzahlen (Integers) verarbeitet werden. Für das Modell existiert somit kein Text, sondern nur diese Zahlenreihe, wobei ein [[#Token]] im Englischen etwa 4 Zeichen bzw. 0,75 Wörtern entspricht. Da das Modell diese Tokens als unteilbare Einheiten wahrnimmt, erklärt diese Architektur auch, warum LLMs paradoxerweise bei einfachen Aufgaben wie dem Buchstabieren oder Zeichenzählen oft scheitern – sie „sehen" das Wort als ganzen Block und nicht die einzelnen Buchstaben darin.

* Karpathy, Andrej. „Let's build the GPT Tokenizer". _YouTube_, 17. Januar 2024. [https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE).

## Tokenizer
id: tokenizer
en: Tokenizer
tags: fundamentals, architecture
level: basic

Ein deterministischer Algorithmus, der als Übersetzer fungiert, um menschenlesbaren Text in eine Sequenz von Ganzzahlen (Integers) umzuwandeln, da neuronale Netze keine Buchstaben, sondern nur mathematische Werte verarbeiten können. Technisch basiert dies meist auf **Byte Pair Encoding (BPE)**: Der Tokenizer analysiert rohe Bytes und verschmilzt iterativ die häufigsten Zeichenpaare zu neuen, größeren Einheiten, bis ein festes Vokabular erreicht ist. Diese [[#Token]]s bilden die unteilbaren „Atome" des Modells; da das Modell den Inhalt eines Tokens (z. B. ein Wortteil) nicht mehr in einzelne Buchstaben zerlegen kann, ist der Tokenizer oft die verborgene Ursache für Probleme bei Aufgaben wie Buchstabieren oder Zeichenzählen.

* Karpathy, Andrej. „Let's build the GPT Tokenizer". _YouTube_, 17. Januar 2024. [https://www.youtube.com/watch?v=zduSFxRajkE](https://www.youtube.com/watch?v=zduSFxRajkE).

## Embedding
id: embedding
en: Embedding
tags: architecture, ai-engineering
level: intermediate

Eine mathematische Repräsentation von [[#Token]] oder Textabschnitten als Vektoren in einem hochdimensionalen Raum. In diesem Raum liegen inhaltlich ähnliche Begriffe (z. B. „König" und „Kaiser") geometrisch nah beieinander, wodurch das Modell semantische Beziehungen berechnen kann.

* Mikolov, Tomas, et al. „Efficient Estimation of Word Representations in Vector Space". _arXiv preprint_, 2013. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781).

## Next Token Prediction
id: next-token-prediction
en: Next Token Prediction
tags: fundamentals, training
level: basic

Next Token Prediction bezeichnet das fundamentale Funktionsprinzip autoregressiver Sprachmodelle, bei dem auf Basis einer Sequenz vorangegangener [[#Token]] die Wahrscheinlichkeitsverteilung für das unmittelbar folgende Token ermittelt wird ($P(w_t | w_{1:t-1})$). Dieses probabilistische Verfahren dient sowohl im Pre-Training als Lernaufgabe (Task) zur Erfassung sprachlicher und inhaltlicher Muster als auch während der Inferenz zur schrittweisen Generierung neuer Texte.

* Bengio, Yoshua, et al. „A Neural Probabilistic Language Model". _Journal of Machine Learning Research_, Bd. 3, 2003, S. 1137–1155. [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

## Retrieval Augmented Generation (RAG)
id: rag
en: RAG (Retrieval Augmented Generation)
tags: ai-engineering
level: intermediate

RAG bezeichnet einen Ansatz, der generative Sprachmodelle mit einem externen Informationsabrufsystem (Retriever) koppelt. Bei diesem Verfahren generiert das Modell Antworten nicht ausschließlich aus den während des Trainings gespeicherten internen Parametern, sondern ruft zunächst relevante Dokumente aus einer externen Wissensbasis (z. B. eine Vektor-Datenbank) ab. Diese abgerufenen Textpassagen werden als zusätzlicher Kontext in den Generierungsprozess eingespeist. Durch diese Methode lässt sich die faktische Genauigkeit der generierten Texte erhöhen und die Neigung zu Halluzinationen verringern. Zudem ermöglicht der Ansatz die Aktualisierung des verfügbaren Wissens durch den einfachen Austausch des Dokumentenindex, ohne dass das neuronale Netz neu trainiert werden muss.

* Lewis, Patrick, et al. „Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". _Advances in Neural Information Processing Systems_, Bd. 33, 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401).

## Vektordatenbank
id: vektordatenbank
en: Vector Database
tags: ai-engineering
level: intermediate

Eine spezialisierte Datenbank, die Informationen nicht als Text oder Tabellen, sondern als hochdimensionale Vektoren (Embeddings) speichert. Sie ermöglicht die semantische Suche: Anstatt nach exakten Schlüsselwörtern zu suchen, berechnet die Datenbank die mathematische Distanz (z. B. Kosinus-Ähnlichkeit) zwischen dem Anfrage-Vektor und den gespeicherten Dokumenten-Vektoren. Das effiziente Durchsuchen dieser hochdimensionalen Räume (_Similarity Search_) erfordert spezialisierte Indexierungsstrukturen, um auch in Milliarden von Datensätzen performant zu bleiben. Dies bildet die technologische Grundlage für RAG-Systeme, da es das schnelle Auffinden von inhaltlich relevantem Kontext aus riesigen Datenmengen ermöglicht.

* Johnson, Jeff, Matthijs Douze, und Hervé Jégou. „Billion-scale similarity search with GPUs". _IEEE Transactions on Big Data_ 7, Nr. 3 (2019): 535–47. [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734).

## Tool Use und Function Calling
id: tool-use
en: Tool Use / Function Calling
tags: agents, ai-engineering
level: intermediate

Die Fähigkeit eines Modells, zu erkennen, dass eine Anfrage externe Werkzeuge erfordert (z. B. Taschenrechner, Wetter-API, Datenbankabfrage), und daraufhin strukturierte Befehle (meist JSON) zu generieren, die von einer Softwareumgebung ausgeführt werden können. Das Ergebnis der Ausführung wird dem Modell zurückgegeben, um die finale Antwort zu formulieren.

* Schick, Timo, und Jane Dwivedi-Yu. „Toolformer: Language Models Can Teach Themselves to Use Tools". _Advances in Neural Information Processing Systems_, Bd. 36, 2023. [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761).

## Model Context Protocol (MCP)
id: mcp
en: Model Context Protocol
tags: ai-engineering, agents
level: intermediate

Ist ein offener Standard, der als universelle Schnittstelle dient, um KI-Assistenten sicher und nahtlos mit externen Datenquellen – wie Content-Repositories, Business-Tools und Entwicklungsumgebungen – zu verbinden. Anstatt für jedes System eine individuelle, fragmentierte Integration entwickeln zu müssen, bietet MCP eine standardisierte Architektur, durch die KI-Modelle direkten Zugriff auf relevante, isolierte Daten erhalten, um so präzisere und kontextbezogenere Antworten liefern zu können.

* Anthropic. „Introducing the Model Context Protocol". _Anthropic News_, 25. November 2024. [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol).

## AI Agent
id: ai-agent
en: AI Agent
tags: agents
level: intermediate

Ein autonomes System, das seine Umgebung wahrnimmt und proaktiv handelt, um definierte Ziele zu erreichen. Im Gegensatz zu passiven Modellen nutzt ein Agent ein LLM als zentrale Reasoning-Einheit, um mehrstufige Pläne zu erstellen und externe Werkzeuge (Tools) oder APIs zur Ausführung zu verwenden. Der Kernprozess ist eine kontinuierliche Schleife aus Beobachtung, Entscheidung und Handlung.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, und Manoj Karkee. „AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (September 2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).

## Agentic AI
id: agentic-ai
en: Agentic AI
tags: agents
level: intermediate

Ein Paradigma in der KI-Entwicklung, das den Grad der Handlungsautonomie (_Agency_) eines Systems beschreibt. Es bezeichnet den Übergang von generativer KI, die lediglich Inhalte erstellt, zu Systemen, die als aktive Problemlöser fungieren. Agentic AI zeichnet sich durch die Fähigkeit aus, komplexe Aufgaben selbstständig in Teilschritte zu zerlegen, die eigenen Ergebnisse zu überprüfen (Self-Reflection) und den Lösungsweg bei Fehlern dynamisch anzupassen.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, und Manoj Karkee. „AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (September 2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).

## Multi-Agent Systems
id: multi-agent-systems
en: Multi-Agent Systems
tags: agents
level: advanced

Systeme, in denen mehrere spezialisierte KI-Agenten miteinander interagieren (kooperieren, debattieren oder konkurrieren), um komplexe Probleme zu lösen. Durch Rollenteilung (z. B. ein Coder, ein Reviewer) können oft bessere Ergebnisse erzielt werden als durch einen einzelnen, monolithischen Agenten.

* Li, Guohao, et al. „CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society". _Advances in Neural Information Processing Systems_, Bd. 36, 2023. [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760).

## World Models
id: world-models
en: World Models
tags: agents, fundamentals
level: advanced

Ein World Model bezeichnet ein generatives KI-System, das eine komprimierte und abstrakte Repräsentation seiner physischen Umgebung erlernt, um deren Dynamik sowie zukünftige Zustände präzise vorherzusagen. Technisch realisiert sich dieses Konzept meist durch eine visuelle Komponente zur Datenreduktion in einen latenten Raum sowie eine zeitliche Komponente zur Simulation kommender Ereignisse auf Basis eigener Aktionen. Diese Architektur ermöglicht es einem Agenten oder Roboter, potenzielle Handlungsfolgen rein mental durchzuspielen und komplexe Pläne zu entwerfen, ohne jeden Schritt riskant in der realen Welt ausprobieren zu müssen. Es fungiert somit als interner Simulator, der die bloße Reaktion auf Reize durch vorausschauendes Planen ersetzt und maschinellen Systemen eine funktionale Intuition für Kausalität und physikalische Gesetzmäßigkeiten verleiht.

* Ha, David, und Jürgen Schmidhuber. „World Models". _arXiv preprint_, 2018. [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

## Halluzinationen (Konfabulationen)
id: halluzinationen
en: Hallucinations / Confabulations
tags: safety, fundamentals
level: basic

Das Generieren von Inhalten, die grammatikalisch und semantisch plausibel klingen, aber faktisch falsch sind oder nicht auf den Trainingsdaten/Quellen basieren. Der Begriff „Konfabulation" wird zunehmend bevorzugt (z. B. von Geoffrey Hinton), da er den Prozess des „Lückenfüllens" ohne Realitätsbezug treffender beschreibt als eine Wahrnehmungsstörung.

* Ji, Ziwei, et al. „Survey of Hallucination in Natural Language Generation". _ACM Computing Surveys_, Bd. 55, Nr. 12, 2023. [https://arxiv.org/abs/2202.03629](https://arxiv.org/abs/2202.03629).

## Understanding (Verstehen)
id: understanding
en: Understanding
tags: fundamentals, safety
level: advanced

Ein hochumstrittener Begriff in der KI-Forschung. Während LLMs eine hohe _funktionale Kompetenz_ (Output ist korrekt) zeigen, bestreiten Kritiker, dass sie eine _formale Kompetenz_ (Verständnis der Bedeutung/Semantik) besitzen. Oft wird argumentiert, dass Modelle nur statistische Papageien sind, die Formen manipulieren, ohne deren Inhalt zu erfassen.

* Bender, Emily M., und Alexander Koller. „Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data". _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, 2020. [https://aclanthology.org/2020.acl-main.463/](https://aclanthology.org/2020.acl-main.463/).

## Bewusstsein und LLM
id: bewusstsein
en: Consciousness in LLMs
tags: fundamentals, safety
level: advanced

Die Debatte, ob Sprachmodelle über subjektives Erleben (_Subjective Experience_ oder _Sentience_) verfügen. Der Philosoph David Chalmers analysiert dies anhand notwendiger Indikatoren (_Feature X_), die aktuellen Modellen fehlen. Er argumentiert, dass heutigen LLMs das Bewusstsein mit hoher Wahrscheinlichkeit fehlt, da sie primär **Feed-Forward-Systeme** ohne Gedächtnisschleifen (**Rekurrenz**) sind und keine robusten **Selbstmodelle** (internes Monitoring) oder eine **einheitliche Agentenschaft** (_Unified Agency_) besitzen. Chalmers skizziert jedoch eine Roadmap zu **LLM+** (erweiterte multimodale Systeme), bei denen durch technische Ergänzungen wie eine _Global Workspace Architektur_ oder verkörperte Interaktion (_Embodiment_) in virtuellen Welten echte Bewusstseinskandidaten entstehen könnten.

* Chalmers, David J. „Could a Large Language Model be Conscious?" arXiv:2303.07103. Preprint, arXiv, 18. August 2024. [https://doi.org/10.48550/arXiv.2303.07103](https://doi.org/10.48550/arXiv.2303.07103).

## AGI (Artificial General Intelligence)
id: agi
en: AGI (Artificial General Intelligence)
tags: fundamentals
level: intermediate

Der Begriff ist umstritten und nicht einheitlich definiert. Nach Bennett (2025) ist AGI kein reines Software-Konstrukt, sondern ein physisch verankertes Gesamtsystem (Hardware und Software), das die Autonomie und Anpassungsfähigkeit eines „künstlichen Wissenschaftlers" besitzt. Anstatt Intelligenz lediglich über die menschliche Leistung bei bekannten Aufgaben zu definieren, bestimmt Bennett AGI als die Fähigkeit, in einer breiten Palette unbekannter Umgebungen Ziele zu erreichen, wobei der wahre Gradmesser nicht die reine Rechenkraft, sondern die **Sample-Effizienz** (Lernen aus wenigen Daten) und die **Energieeffizienz** bei der Adaptation an neue Probleme ist.

* Bennett, Michael Timothy. _What the F*ck Is Artificial General Intelligence?_ Bd. 16057. 2026. [https://doi.org/10.1007/978-3-032-00686-8_4](https://doi.org/10.1007/978-3-032-00686-8_4).

## Parameter
id: parameter
en: Parameters
tags: fundamentals, architecture
level: basic

Bezeichnen die internen Konfigurationsvariablen (primär Gewichte und Biases) eines neuronalen Netzes, die während des Trainingsprozesses durch mathematische Optimierung (Backpropagation) gelernt und justiert werden, um den Vorhersagefehler zu minimieren. Sie repräsentieren die Stärke der Verbindungen zwischen den künstlichen Neuronen und speichern somit das gesamte extrahierte Wissen sowie die Fähigkeiten des Modells in Form von gigantischen Zahlenmatrizen. Während der [[#Inferenz]] bleiben diese Werte statisch; die schiere Anzahl der Parameter (oft in Milliarden gemessen) gilt gemäß den [[#Scaling Laws]] als primärer Indikator für die potenzielle Kapazität eines Modells, bestimmt jedoch gleichzeitig direkt den Bedarf an Rechenleistung und Grafikspeicher (VRAM).

* Goodfellow, Ian, Yoshua Bengio, und Aaron Courville. _Deep Learning_. MIT Press, 2016. [http://www.deeplearningbook.org](http://www.deeplearningbook.org/).

## Mixture of Experts (MoE)
id: moe
en: Mixture of Experts
tags: architecture
level: advanced

Eine Modellarchitektur, bei der das neuronale Netz nicht als ein einziger monolithischer Block aktiviert wird, sondern in viele kleine Sub-Netzwerke („Experten") unterteilt ist. Für jedes Token entscheidet ein „Router", welche Experten (meist nur 1-2) aktiviert werden. Dies erlaubt Modelle mit extrem vielen Parametern (z. B. GPT-4, Mixtral), die dennoch schnell und kostengünstig in der [[#Inferenz]] sind, da immer nur ein Bruchteil des Netzes rechnet.

* Fedus, William, Barret Zoph, und Noam Shazeer. „Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity". _Journal of Machine Learning Research_, Bd. 23, 2022. [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961).

## Quantisierung
id: quantisierung
en: Quantization
tags: ai-engineering, architecture
level: intermediate

Ein Verfahren zur Reduktion des Speicherbedarfs und der Rechenlast eines LLMs, indem die Genauigkeit der Modellgewichte reduziert wird (z. B. von 16-Bit-Gleitkommazahlen auf 4-Bit-Ganzzahlen). Dies ermöglicht es, riesige Modelle auf consumer-Hardware (lokalen Laptops/GPUs) laufen zu lassen, oft mit nur minimalem Qualitätsverlust.

* Dettmers, Tim, et al. „LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale". _Advances in Neural Information Processing Systems_, Bd. 35, 2022. [https://arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339).

## Multimodalität (LMM)
id: multimodalitaet
en: Multimodality / LMM
tags: fundamentals, architecture
level: basic

Bezeichnet die Fähigkeit von KI-Modellen (Large Multimodal Models), verschiedene Datentypen – wie Text, Bilder, Audio und Video – nicht nur isoliert zu verarbeiten, sondern in einem gemeinsamen semantischen Raum zu verstehen. Anders als frühere Systeme, die verschiedene Modelle kombinierten (z. B. eine Bilderkennungs-KI, die Text an ein LLM übergibt), sind LMMs „nativ" multimodal trainiert. Sie können beispielsweise ein Meme analysieren, indem sie gleichzeitig den visuellen Inhalt und die kulturelle Bedeutung des Textes erfassen und deren ironisches Zusammenspiel verstehen.

## RLHF vs. RLAIF
id: rlaif
en: RLAIF
tags: training
level: advanced

Reinforcement Learning from Human Feedback (RLHF) nutzt menschliche Bewertungen, um das Modell zu belohnen oder zu bestrafen. **Reinforcement Learning from AI Feedback (RLAIF)** automatisiert diesen Prozess, indem ein starkes KI-Modell die Ausgaben eines anderen Modells bewertet. RLAIF ist entscheidend für die Skalierung, da menschliches Feedback teuer und langsam ist.

* **RLHF:** Christiano, Paul F., et al. „Deep Reinforcement Learning from Human Preferences". _Advances in Neural Information Processing Systems_, Bd. 30, 2017. [https://arxiv.org/abs/1706.03741](https://arxiv.org/abs/1706.03741).
* **RLAIF:** Lee, Harrison, et al. „RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback". _arXiv preprint_, 2023. [https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267)

## Prompt Injection
id: prompt-injection
en: Prompt Injection
tags: safety
level: intermediate

Prompt Injection bezeichnet eine Angriffstechnik auf Large Language Models (LLMs), bei der gezielt konstruierte Eingaben verwendet werden, um die vom Entwickler vorgegebenen Systemanweisungen zu überschreiben und die ursprüngliche Zielsetzung des Modells zu kapern. Indem der Angreifer die Flexibilität der natürlichen Spracheingabe ausnutzt, wird das Modell dazu gebracht, seine eigentliche Aufgabe zu ignorieren und stattdessen die im Prompt versteckten Befehle auszuführen, was oft zum Leaken interner Instruktionen oder zur Manipulation der Anwendungslogik führt.

* Greshake, Kai, et al. „Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection". _arXiv preprint_, 2023. [https://arxiv.org/abs/2302.12173](https://arxiv.org/abs/2302.12173).

## Jailbreak
id: jailbreak
en: Jailbreak
tags: safety
level: intermediate

Beschreibt eine spezielle Form des Adversarial Prompting, die primär darauf abzielt, die implementierten Sicherheitsfilter und Inhaltsbeschränkungen (Content Restrictions) eines Modells zu umgehen. Durch das Einbetten von Anfragen in hypothetische Szenarien, komplexe Rollenspiele oder simulierte „Entwicklermodi" wird das Modell dazu manipuliert, seine Sicherheitsrichtlinien temporär außer Kraft zu setzen und unzensierte Ausgaben zu generieren, die unter normalen Umständen blockiert würden.

* Greshake, Kai, et al. „Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection". _arXiv preprint_, 2023. [https://arxiv.org/abs/2302.12173](https://arxiv.org/abs/2302.12173).

## Model Collapse
id: model-collapse
en: Model Collapse
tags: training, safety
level: advanced

Model Collapse bezeichnet einen irreversiblen degenerativen Lernprozess, der auftritt, wenn generative Modelle über mehrere Generationen hinweg primär auf dem Output von Vorgängermodellen – also auf [[#Synthetische Daten]] – trainiert werden. Da statistische Modelle dazu neigen, wahrscheinliche Muster zu verstärken und seltene Randereignisse („Tails") zu glätten, führt diese rekursive Rückkopplung dazu, dass die Varianz der ursprünglichen Verteilung schrittweise verloren geht. Das Modell „vergisst" die wahre Datenverteilung, was zunächst zum Verschwinden seltener Nuancen führt und im Endstadium in einer stark vereinfachten, repetitiven Darstellung der Realität mit minimaler Diversität mündet. Dies unterstreicht die Notwendigkeit, synthetische Trainingsdaten stets mit originalen, menschengemachten Daten zu kuratieren.

* Shumailov, Ilia, et al. „The Curse of Recursion: Training on Generated Data Makes Models Forget". _Nature_, 2024. [https://arxiv.org/abs/2305.17493](https://arxiv.org/abs/2305.17493).

## Test-Time Compute
id: test-time-compute
en: Test-Time Compute
tags: architecture, prompting
level: advanced

Ein Paradigma, bei dem die Rechenleistung nicht nur massiv in das Training des Modells investiert wird, sondern gezielt während der [[#Inferenz]] („Testzeit") eingesetzt wird. Anstatt wie klassische LLMs sofort das nächste Token vorherzusagen (System-1-Denken: intuitiv, schnell), nutzt das Modell zusätzliche Rechenzeit, um intern verschiedene Lösungswege zu simulieren, Fehler zu korrigieren und Schritte zu verifizieren (System-2-Denken: analytisch, langsam), bevor es eine Antwort ausgibt. Dieser Ansatz ist der Kern von „Reasoning Models" wie OpenAI o1.

* OpenAI o1 System Card. https://openai.com/index/learning-to-reason-with-llms/

## Test-Time Adaptation
id: test-time-adaptation
en: Test-Time Adaptation
tags: architecture, training
level: advanced

Ist der Prozess, bei dem ein KI-Modell nicht einfach nur sein eingefrorenes Trainingswissen abruft, sondern während der [[#Inferenz]] (Testzeit) zusätzliche Rechenleistung aufwendet, um sich spezifisch auf das gerade vorliegende Problem anzupassen.

* Test-Time Adaptation: A New Frontier in AI. https://youtu.be/C6sSs6NgANo

# Ressourcen

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary
