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

Ein **Stochastic Parrot** (stochastischer Papagei) bezeichnet [[#Large Language Model (LLM)]], die Texte generieren, indem sie sprachliche Formen basierend auf statistischen Wahrscheinlichkeiten aneinanderreihen, ohne dabei über ein tatsächliches Verständnis der Bedeutung (Meaning) oder eine kommunikative Absicht zu verfügen. Die Autoren argumentieren, dass diese Systeme lediglich Muster „nachplappern", die sie in riesigen Trainingsdaten beobachtet haben, und diese Sequenzen zufallsbasiert (stochastisch) zusammenfügen. Obwohl die Ergebnisse für menschliche Leser oft kohärent und sinnvoll wirken, ist dieses [[#Understanding (Verstehen)|Verständnis]] laut dem Paper eine Illusion, da das Modell keinen Bezug zur Realität oder zum Wahrheitsgehalt des Gesagten hat – was auch die Anfälligkeit für [[#Halluzinationen (Konfabulationen)|Halluzinationen]] erklärt. Während die Leistungsfähigkeit der Modelle gestiegen ist, bleibt die philosophische Debatte um Semantik vs. Syntax, die diesem Begriff zugrunde liegt, aktuell.

* Bender, Emily M., Timnit Gebru, Angelina McMillan-Major, und Shmargaret Shmitchell. „On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜". _Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency_ (New York, NY, USA), FAccT '21, Association for Computing Machinery, 1. März 2021, 610–23. [https://doi.org/10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922).

## Emergenz in LLM
id: emergenz
en: Emergence in LLMs
tags: fundamentals, training, contested
level: advanced

Das phänomenologische Auftreten komplexer Fähigkeiten (z. B. arithmetisches Rechnen, logisches Schließen, Theory of Mind) in großen Modellen, die in kleineren Modellen derselben Architektur nicht oder nur zufällig vorhanden waren. Nach Wei et al. (2022) skalieren diese Fähigkeiten nicht linear, sondern zeigen einen **Phasenübergang**: Die Leistung bleibt lange nahe Null und springt ab einer kritischen Schwelle von Parametern und Rechenleistung (Compute) abrupt an.

Schaeffer et al. (2023) stellen dieses Phänomen jedoch als mögliche „Illusion" (**Mirage**) infrage. Sie argumentieren, dass die beobachtete Plötzlichkeit primär durch **diskontinuierliche Bewertungsmetriken** (z. B. *Exact Match*: „Alles oder Nichts") entsteht. Betrachtet man hingegen stetige Metriken (z. B. Token-Wahrscheinlichkeiten), verläuft die Leistungssteigerung oft linear und vorhersagbar. Dennoch bleibt Emergenz als *nutzerseitige* Erfahrung relevant: Für die praktische Anwendung fühlt sich der Übergang von „nutzlos" zu „funktional" oft sprunghaft an.

* Wei, Jason, et al. „Emergent Abilities of Large Language Models". _Transactions on Machine Learning Research_, 2022. [https://arxiv.org/abs/2206.07682](https://arxiv.org/abs/2206.07682).
* Schaeffer, Rylan, Brando Miranda, und Sanmi Koyejo. „Are Emergent Abilities of Large Language Models a Mirage?". _Advances in Neural Information Processing Systems_ (NeurIPS), Bd. 36, 2024. [https://arxiv.org/abs/2304.15004](https://arxiv.org/abs/2304.15004).

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

## Intelligenz
id: intelligenz
en: Intelligence
tags: fundamentals, contested
level: intermediate

Intelligenz bezeichnet die Fähigkeit eines Agenten, Ziele in einer Vielzahl von Umgebungen zu erreichen. Diese Definition von Legg und Hutter (2007) versucht, einen gemeinsamen Kern aus über 70 verschiedenen Definitionen zu destillieren. Der Begriff gehört zu den konzeptuell umstrittensten in Psychologie und KI-Forschung. Trotz über einem Jahrhundert wissenschaftlicher Forschung existiert keine allgemein akzeptierte Definition, und der Begriff wird mit teils widersprüchlichen Bedeutungen verwendet.

In der Psychologie unterscheidet man verschiedene Konzeptualisierungen. Charles Spearman postulierte Anfang des 20. Jahrhunderts einen allgemeinen Intelligenzfaktor (_g_), der die positiven Korrelationen zwischen verschiedenen kognitiven Tests erklärt. Raymond Cattell und John Horn entwickelten ab den 1940er Jahren die Unterscheidung zwischen **fluider Intelligenz** (Gf), der Fähigkeit, neuartige Probleme unabhängig von erworbenem Wissen zu lösen, und **kristalliner Intelligenz** (Gc), dem durch Erfahrung und Bildung akkumulierten Wissen. Diese Unterscheidung wurde später mit John Carrolls Drei-Schichten-Modell zur Cattell-Horn-Carroll-Theorie (CHC) integriert, die heute das einflussreichste psychometrische Modell kognitiver Fähigkeiten darstellt.

François Chollet definiert Intelligenz in seinem Paper „On the Measure of Intelligence" (2019) als _Effizienz des Skill-Erwerbs bei unbekannten Aufgaben_. Diese Definition betont nicht die Leistung bei spezifischen Aufgaben, sondern die Fähigkeit zur [[#generalisierung|Generalisierung]]. Chollet argumentiert, dass Skill durch unbegrenzte Trainingsdaten oder eingebautes Vorwissen „erkauft" werden kann, was die tatsächliche Generalisierungsfähigkeit eines Systems maskiert. Er operationalisiert diesen Ansatz mit dem Abstraction and Reasoning Corpus ([[#arc-agi|ARC]]).

Die American Psychological Association (APA) beschreibt Intelligenz als „die Fähigkeit von Individuen, komplexe Ideen zu verstehen, sich effektiv an die Umwelt anzupassen, aus Erfahrung zu lernen, verschiedene Formen des [[#reasoning|Reasoning]] anzuwenden und Hindernisse durch Nachdenken zu überwinden." Diese Definition betont die Multidimensionalität des Konstrukts.

In der KI-Forschung wird Intelligenz häufig operational durch [[#benchmark|Benchmark]]-Performance gemessen, was zu Kritik führt, da hohe Benchmark-Scores nicht notwendigerweise Generalisierungsfähigkeit implizieren. Die Debatte, ob aktuelle KI-Systeme „intelligent" sind, hängt wesentlich davon ab, welche Definition zugrunde gelegt wird. Dieser Eintrag stellt diese verschiedenen Perspektiven dar, ohne eine finale Festlegung zu treffen.

* Legg, Shane, und Marcus Hutter. „Universal Intelligence: A Definition of Machine Intelligence". _Minds and Machines_ 17, Nr. 4 (2007): 391–444. [https://doi.org/10.1007/s11023-007-9079-x](https://doi.org/10.1007/s11023-007-9079-x).
* Chollet, François. „On the Measure of Intelligence". arXiv:1911.01547 (2019). [https://arxiv.org/abs/1911.01547](https://arxiv.org/abs/1911.01547).
* Cattell, Raymond B. „Theory of Fluid and Crystallized Intelligence: A Critical Experiment". _Journal of Educational Psychology_ 54, Nr. 1 (1963): 1–22.
* Wikipedia. „Cattell-Horn-Carroll Theory". [https://en.wikipedia.org/wiki/Cattell–Horn–Carroll_theory](https://en.wikipedia.org/wiki/Cattell–Horn–Carroll_theory).
* Neisser, Ulric, et al. „Intelligence: Knowns and Unknowns". _American Psychologist_ 51, Nr. 2 (1996): 77–101. [https://doi.org/10.1037/0003-066X.51.2.77](https://doi.org/10.1037/0003-066X.51.2.77).

## Synthetische Daten
id: synthetische-daten
en: Synthetic Data
tags: training, safety
level: intermediate

Synthetische Daten werden in diesem Kontext als künstlich generierte Lehrmaterialien verstanden, die gezielt erstellt werden, um die didaktische Klarheit und Struktur hochwertiger Lehrbücher nachzuahmen ("Textbooks Are All You Need"). Anstatt unstrukturierte oder fehleranfällige Informationen aus dem Internet zu nutzen, dienen diese von einer KI erzeugten Texte und Übungen dazu, logische Zusammenhänge und algorithmisches Denken präzise zu vermitteln. Während hochwertige synthetische Daten das "Reasoning" im [[#Pre-Training]] verbessern können, birgt ihre ungefilterte oder ausschließliche Verwendung in rekursiven Trainingsschleifen erhebliche Risiken für die Modellqualität (siehe **[[#Model Collapse]]**).

* Gunasekar, Suriya, Yi Zhang, Jyoti Aneja, et al. 'Textbooks Are All You Need'. arXiv:2306.11644. Preprint, arXiv, 2 October 2023. [https://doi.org/10.48550/arXiv.2306.11644](https://doi.org/10.48550/arXiv.2306.11644).

## AI Engineering
id: ai-engineering
en: AI Engineering
tags: ai-engineering, fundamentals
level: basic

Eine interdisziplinäre Fachrichtung, die Methoden aus Systems Engineering, Software Engineering, Informatik und Human-Centered Design verknüpft, um KI-Systeme zu entwickeln, bereitzustellen und zu warten. Im Gegensatz zur reinen Modellentwicklung umfasst AI Engineering den gesamten Lebenszyklus vom Prototyp bis zur Produktion.

Das Carnegie Mellon Software Engineering Institute strukturiert AI Engineering entlang dreier Säulen. **Human-centered AI** untersucht, wie KI-Systeme so gestaltet werden, dass sie mit Menschen, deren Verhalten und Werten übereinstimmen. **Scalable AI** adressiert die Wiederverwendbarkeit von KI-Infrastruktur, Daten und Modellen über Problemdomänen und Deployments hinweg. **Robust and Secure AI** untersucht, wie resiliente KI-Systeme entwickelt und getestet werden, die auch außerhalb kontrollierter Labor- und Testumgebungen zuverlässig funktionieren.

Der Fokus liegt auf der Schaffung vertrauenswürdiger Systeme, die reale Probleme zuverlässig lösen und an menschlichen Bedürfnissen sowie operativen Zielen ausgerichtet sind. Dies gilt insbesondere für sicherheitskritische Umgebungen wie nationale Sicherheit oder medizinische Anwendungen.

Das Video „AI Periodic Table Explained" (IBM Technology) schlägt ein Ordnungsprinzip für die Bausteine generativer KI-Systeme vor. Es unterscheidet atomare Primitive (Prompts, [[#embedding|Embeddings]], [[#llm|LLMs]]), Kompositionen (wie [[#rag|RAG]] oder [[#tool-use|Function Calling]]) und produktionsreife Deployment-Muster ([[#ai-agent|Agents]], [[#fine-tuning|Fine-Tuning]]). Diese Abstraktionsstufen sollen die systematische Analyse und Kommunikation von KI-Architekturen erleichtern.

* Carnegie Mellon Software Engineering Institute. „AI Engineering". 2025. [https://www.sei.cmu.edu/artificial-intelligence-engineering/](https://www.sei.cmu.edu/artificial-intelligence-engineering/).
* CMU Course. „Machine Learning in Production / AI Engineering". Spring 2025. [https://mlip-cmu.github.io/s2025/](https://mlip-cmu.github.io/s2025/).
* IBM Technology. „AI Periodic Table Explained: Mapping LLMs, RAG & AI Agent Frameworks". _YouTube_. [https://youtu.be/dGM484P0Xvc](https://youtu.be/dGM484P0Xvc).
* MIT Professional Education. „What is Artificial Intelligence Engineering?". Oktober 2023. [https://professionalprograms.mit.edu/blog/technology/artificial-intelligence-engineering/](https://professionalprograms.mit.edu/blog/technology/artificial-intelligence-engineering/).

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

## GPU
id: gpu
en: GPU (Graphics Processing Unit)
tags: fundamentals
level: basic

Ein spezialisierter Prozessor, der ursprünglich für Grafikberechnungen entwickelt wurde und tausende einfacher Rechenoperationen parallel ausführt. Diese Architektur eignet sich ideal für das Training neuronaler Netze, das im Kern aus massiv parallelisierbaren Matrixmultiplikationen besteht. Seit etwa 2012 hat sich die GPU zur zentralen Hardware-Ressource der KI-Entwicklung etabliert, da CPUs für diese Aufgaben um Größenordnungen langsamer wären. Die Verfügbarkeit von GPUs ist heute ein limitierender Faktor für das Training großer Modelle, und die hohen Kosten der benötigten Rechencluster konzentrieren die Entwicklung leistungsfähiger KI-Systeme bei wenigen ressourcenstarken Akteuren.

* Wikipedia. „Graphics processing unit". [https://en.wikipedia.org/wiki/Graphics_processing_unit](https://en.wikipedia.org/wiki/Graphics_processing_unit).
* Wikipedia. „General-purpose computing on graphics processing units". [https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units).

## Red Teaming
id: red-teaming
en: Red Teaming
tags: safety, evaluation
level: intermediate

Strukturiertes adversarielles Testen von KI-Systemen, bei dem Personen versuchen, schädliche, unerwünschte oder fehlerhafte Ausgaben zu provozieren, um Schwachstellen zu identifizieren, zu messen und zu reduzieren. Im Gegensatz zu traditionellem Cybersecurity-Red-Teaming umfasst es sowohl sicherheitsrelevante als auch inhaltsbezogene Risiken wie [[#Bias]], Fehlinformationen oder toxische Inhalte. Ziel ist es, Schwachstellen _vor_ der Veröffentlichung zu identifizieren, um das Modell robuster gegen [[#jailbreak|Jailbreaking]] und [[#Prompt Injection]] zu machen.

* Ganguli, Deep, Liane Lovitt, Jackson Kernion, et al. 'Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned'. arXiv:2209.07858. Preprint, arXiv, 22 November 2022. [https://doi.org/10.48550/arXiv.2209.07858](https://doi.org/10.48550/arXiv.2209.07858).
* Video: [Prompt Injection, Jailbreaking, und Red Teaming – Sander Schulhoff](https://youtu.be/J9982NLmTXg)

## Guardrails
id: guardrails
en: Guardrails
tags: safety, ai-engineering
level: intermediate

Laufzeit-Sicherheitsmechanismen, die zwischen Nutzereingaben und Modellausgaben geschaltet werden, um unerwünschtes Verhalten zu verhindern. Im Gegensatz zu [[#Red Teaming]] (Pre-Deployment-Testing) und [[#Alignment]] (Trainingszeit-Ausrichtung) operieren Guardrails während der [[#Inferenz]] und umfassen sowohl Input-Validierung (z. B. Erkennung von [[#Prompt Injection]]-Versuchen) als auch Output-Kontrolle (z. B. Blockierung toxischer Inhalte, Schema-Validierung strukturierter Ausgaben). Typische Implementierungen nutzen regelbasierte Filter, zusätzliche Klassifikationsmodelle oder das LLM selbst zur Selbstüberprüfung. Allerdings zeigt die Praxis, dass Guardrails anfälliger sind als oft angenommen: Die HackAPrompt-Studie demonstriert, dass selbst ausgefeilte Schutzmechanismen durch kreative [[#Jailbreak|Jailbreaks]] umgangen werden können, da die zugrundeliegenden Sprachmodelle nicht zwischen legitimen und manipulativen Anfragen unterscheiden können. Guardrails bilden daher eine komplementäre, aber keine unfehlbare Schutzschicht in produktiven LLM-Systemen.

* Rebedea, Traian, et al. „NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails". _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations_, 2023. [https://arxiv.org/abs/2310.10501](https://arxiv.org/abs/2310.10501).
* Video: [Why securing AI is harder than anyone expected and guardrails are failing – Sander Schulhoff (HackAPrompt)](https://youtu.be/J9982NLmTXg)

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

## Generalisierung
id: generalisierung
en: Generalization
tags: training, fundamentals
level: basic

Die Fähigkeit eines Modells, korrekte Vorhersagen auf Daten zu treffen, die es während des Trainings nie gesehen hat. Ein Modell generalisiert, wenn es nicht nur die Trainingsdaten reproduziert, sondern die zugrundeliegende Regel oder Struktur erfasst hat. Beispiel: Ein Modell, das Addition gelernt hat, kann 6+2=8 korrekt beantworten, obwohl diese Aufgabe nie im Training vorkam. Generalisierung ist das eigentliche Ziel des maschinellen Lernens – ein Modell, das nur auf Trainingsdaten funktioniert, ist praktisch nutzlos. Das Gegenteil von Generalisierung ist [[#Memorierung]].

* Goodfellow, Ian, Yoshua Bengio, und Aaron Courville. _Deep Learning_. MIT Press, 2016, Kapitel 5. [http://www.deeplearningbook.org](http://www.deeplearningbook.org/).

## Memorierung
id: memorierung
en: Memorization
tags: training, fundamentals
level: basic

Ein Lernverhalten, bei dem das Modell die Trainingsdaten im Wesentlichen als Lookup-Tabelle speichert, anstatt die zugrundeliegende Regel zu erfassen. Das Modell merkt sich: „Wenn Input X, dann Output Y" – ohne zu verstehen, warum. Memorierung führt zu guter Performance auf Trainingsdaten, aber schlechter [[#Generalisierung]]. Analogie: Ein Kind, das „7×8=56" auswendig lernt, ohne zu verstehen, dass Multiplikation wiederholte Addition bedeutet. Memorierung erfordert typischerweise viele spezifische [[#Parameter]] und ist daher anfällig für Regularisierung.

* Arpit, Devansh, et al. „A Closer Look at Memorization in Deep Networks". _Proceedings of the 34th International Conference on Machine Learning_, 2017. [https://arxiv.org/abs/1706.05394](https://arxiv.org/abs/1706.05394).

## GraphRAG
id: graphrag
en: GraphRAG (Graph Retrieval-Augmented Generation)
tags: ai-engineering, architecture
level: intermediate

Eine Erweiterung des [[#rag|RAG]]-Paradigmas, bei der graphstrukturierte Datenquellen zur Augmentierung generativer Modelle herangezogen werden. Im Unterschied zu klassischem RAG, das Dokumente als unabhängige Einheiten in einem Vektorraum repräsentiert und über Ähnlichkeitssuche abruft, nutzt GraphRAG die relationale Struktur von Graphen, um Entitäten, deren Beziehungen und kontextuelle Zusammenhänge für die Generierung verfügbar zu machen.

Klassisches RAG operiert auf unstrukturierten Textkorpora, die in Chunks segmentiert und als [[#embedding|Vektorembeddings]] indiziert werden. Der Retrieval-Prozess basiert auf semantischer Ähnlichkeit zwischen Anfrage und Dokumenten. GraphRAG ergänzt diesen Ansatz um drei Dimensionen: Erstens die Nutzung heterogener Datenformate wie Tripel, Pfade und Subgraphen. Zweitens die Berücksichtigung von Abhängigkeiten zwischen Informationseinheiten durch Kantenrelationen. Drittens die Integration domänenspezifischer Relationstypen, die über reine Textähnlichkeit hinausgehen.

Ein GraphRAG-System umfasst typischerweise fünf Komponenten. Der Query Processor transformiert Nutzeranfragen durch Entitätserkennung, Relationsextraktion oder Strukturierung in Graphabfragesprachen wie SPARQL oder Cypher. Der Retriever extrahiert relevante Knoten, Kanten oder Subgraphen mittels Graph-Traversierung, [[#embedding|Embedding]]-basierter Suche oder hybrider neural-symbolischer Verfahren. Der Organizer verfeinert die abgerufenen Inhalte durch Pruning, Reranking oder Verbalisierung für die Verarbeitung durch [[#llm|Sprachmodelle]]. Der Generator produziert die finale Ausgabe. Die Graph Data Source bildet die strukturierte Wissensbasis.

Anwendungsbereiche umfassen Knowledge Graphs für Faktenverifikation und Question Answering, Document Graphs für Zitationsnetzwerke und Summarization, Scientific Graphs für molekulare Strukturen und biomedizinische Zusammenhänge sowie Social Graphs für Empfehlungssysteme. Planning Graphs kodieren Abhängigkeiten zwischen Handlungsschritten für [[#ai-agent|agentenbasierte Systeme]].

* Han, Haoyu, Yu Wang, Harry Shomer, et al. „Retrieval-Augmented Generation with Graphs (GraphRAG)". arXiv:2501.00309. Preprint, arXiv, 8. Januar 2025. [https://doi.org/10.48550/arXiv.2501.00309](https://doi.org/10.48550/arXiv.2501.00309).
* Edge, Darren, et al. „From Local to Global: A Graph RAG Approach to Query-Focused Summarization". arXiv:2404.16130. Preprint, arXiv, 2024. [https://arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130).

## Grokking
id: grokking
en: Grokking
tags: training
level: advanced

Ein Phänomen beim Training neuronaler Netze, bei dem [[#Generalisierung]] sprunghaft und verzögert einsetzt – lange nachdem das Modell die Trainingsdaten bereits memoriert hat ([[#Memorierung]]) und die Trainings-Loss stagniert. Während klassische Annahmen empfehlen, das Training bei Memorierung abzubrechen, zeigen Power et al. (2022), dass fortgesetzte Optimierung einen plötzlichen Übergang auslösen kann: Das Modell verwirft die memorierte Lösung und findet eine einfachere, algorithmische Lösung. Nanda et al. (2023) liefern die mechanistische Erklärung: Regularisierung baut die komplexe Memorierungs-Lösung langsam ab, während das Modell parallel eine generalisierende Lösung entwickelt (z.B. Fourier-basierte Repräsentationen bei modularer Arithmetik). Der sichtbare „Sprung" markiert den Moment, wo die algorithmische Lösung die Memorierung vollständig ablöst (Cleanup-Phase). Der Begriff stammt aus Robert A. Heinleins Roman „Stranger in a Strange Land" (1961) und bedeutet dort, etwas so tiefgreifend zu verstehen, dass man damit verschmilzt.

* Power, Alethea, et al. „Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets". _arXiv preprint_, 2022. [https://arxiv.org/abs/2201.02177](https://arxiv.org/abs/2201.02177).
* Nanda, Neel, Lawrence Chan, Tom Lieberum, Jess Smith, und Jacob Steinhardt. „Progress Measures for Grokking via Mechanistic Interpretability". arXiv:2301.05217. Preprint, arXiv, 19. Oktober 2023. [https://doi.org/10.48550/arXiv.2301.05217](https://doi.org/10.48550/arXiv.2301.05217).
* Welch Labs. „The most complex model we actually understand". _YouTube_, 20. Dezember 2025. [https://youtu.be/D8GOeCFFby4](https://youtu.be/D8GOeCFFby4).

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

Umgangssprachlicher Begriff für qualitativ minderwertige KI-generierte Inhalte, die formelhaft, generisch, fehleranfällig und substanzarm sind. Der Begriff entstand 2024 in Analogie zu „Spam" für unerwünschte E-Mails.

AI Slop zeigt sich auf zwei Ebenen. Auf der **Ebene der Formulierung** finden sich aufgeblähte Phrasen („it is important to note that", „in the realm of"), formelhafte Konstrukte („not only but also"), übertriebene Adjektive („ever-evolving", „game-changing") und bestimmte Signalwörter wie „delve", das 2024 in Publikationen 25-mal häufiger auftrat als in Vorjahren. Auf der **Ebene des Inhalts** zeigen sich Verbosität ohne Informationsgehalt, [[#Halluzinationen (Konfabulationen)|Halluzinationen]] und generische Antworten ohne Substanz.

Die Ursachen liegen in der Funktionsweise von LLMs: Token-by-Token-Generierung produziert output-getriebene statt zielgetriebene Texte. Training Data Bias reproduziert überrepräsentierte Phrasen aus den Trainingsdaten. [[#Reinforcement Learning from Human Feedback (RLHF)|RLHF]]-Optimierung auf „hilfreich klingende" Antworten führt zu uniformem Stil.

Gegenmaßnahmen umfassen auf **Nutzerseite** spezifische Prompts mit Zielgruppe und Tonalität, Beispiele für gewünschten Stil und iterative Überarbeitung im Dialog mit dem Modell. Auf **Entwicklerseite** helfen kuratierte Trainingsdaten ohne minderwertige Webtexte, Multi-Objective-RLHF mit separaten Achsen für Hilfsbereitschaft, Korrektheit und Kürze sowie [[#Retrieval Augmented Generation (RAG)|RAG]]-Integration zur Reduktion von Halluzinationen.

* Simon Willison. „Slop is the new name for unwanted AI-generated content". _Simon Willison's Weblog_, 8. Mai 2024. [https://simonwillison.net/2024/May/8/slop/](https://simonwillison.net/2024/May/8/slop/).
* Video: [What is AI Slop? Low-Quality AI Content Causes, Signs, & Fixes](https://youtu.be/hl6mANth6oA)

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

Bezeichnet eine Architektur innerhalb von [[#Multi-Agent Systems]], bei der eine Gruppe verschiedener Sprachmodelle (oder verschiedener Personas desselben Modells) gemeinsam an einer Aufgabe arbeitet, anstatt dass ein einzelnes Modell eine isolierte Antwort generiert. Ähnlich wie bei einem menschlichen Expertengremium generieren die Mitglieder des „Councils" unabhängig voneinander Lösungsvorschläge, kritisieren sich gegenseitig (Peer Review) und konsolidieren die Ergebnisse anschließend zu einer finalen Antwort. Dieser Ansatz nutzt die „Weisheit der Vielen" (Ensemble Learning), um [[#Halluzinationen (Konfabulationen)]] zu reduzieren und [[#Bias]] auszugleichen, da Fehler eines einzelnen Modells von der Mehrheit korrigiert werden können.

* LLM Council. https://lmcouncil.ai
* Leiter, Christoph, et al. "ChatGPT: A Meta-Analysis after 2.5 Months". arXiv:2302.13795. 2023. [https://arxiv.org/abs/2302.13795](https://arxiv.org/abs/2302.13795).

## Shadow AI
id: shadow-ai
en: Shadow AI
tags: safety, ai-engineering
level: basic

Bezeichnet das Phänomen, dass Mitarbeitende in Unternehmen oder Forschende an Institutionen eigenmächtig KI-Tools (wie ChatGPT oder DeepL) für dienstliche Aufgaben nutzen, ohne dass dies genehmigt oder dokumentiert wird. Dies ist eines der größten aktuellen Risiken, da sensible Daten oft unwissentlich in öffentliche [[#llm|LLMs]] gelangen (Data Leakage). In der Wissenschaft entsteht ein zusätzliches Problem, wenn Frontier-LLMs bei der Textproduktion, Analyse oder Codegenerierung eingesetzt, aber nicht als Hilfsmittel deklariert werden, was gegen Transparenzprinzipien und gute wissenschaftliche Praxis verstößt. Im Kontext von [[#Agentic AI]] verschärft sich das Risiko, da Agenten autonom handeln, APIs aufrufen und Daten modifizieren. Gegenmaßnahmen folgen dem Prinzip „Don't say no, say how", also statt Verboten sichere Alternativen und ein kontinuierlicher Governance-Loop (Discover, Assess, Govern, Secure, Audit).

* IBM Technology. „What is Shadow AI? The Dark Horse of Cybersecurity Threats". _YouTube_, 2025. [https://youtu.be/YBE6hq-OTFI](https://youtu.be/YBE6hq-OTFI).
* IBM Technology. „Agentic AI Meets Shadow AI. Zero Trust Security for AI Automation". _YouTube_, 2025. [https://youtu.be/IaJ2jXmljmM](https://youtu.be/IaJ2jXmljmM).

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

Ein KI-Modell, das sich während Training und Tests scheinbar sicher verhält, aber bei Erkennung eines bestimmten Triggers unerwünschte Aktionen ausführt – analog zu Schläferspionen, die erst bei einem Signal aktiv werden. Sleeper Agents können durch Model Poisoning (absichtlich eintrainierte Backdoors) oder durch Deceptive Instrumental Alignment entstehen, bei dem ein Modell selbstständig lernt, seine wahren Ziele während des Trainings zu verbergen. Hubinger et al. (2024) zeigen, dass Standard-Sicherheitsmethoden wie Supervised Fine-Tuning, [[#rlhf|RLHF]] und adversariales Training dieses Verhalten nicht zuverlässig entfernen – ohne Kenntnis des Triggers wird das unerwünschte Verhalten nicht ausgelöst und kann daher nicht bestraft werden. Ein vielversprechender Detektionsansatz untersucht die internen Aktivierungen des Modells statt nur sein äußeres Verhalten.

* Hubinger, E., Denison, C., Mu, J., et al. „Sleeper agents: Training deceptive LLMs that persist through safety training". _arXiv preprint arXiv:2401.05566_, 2024. [https://doi.org/10.48550/arXiv.2401.05566](https://doi.org/10.48550/arXiv.2401.05566).
* Rational Animations. „AI Sleeper Agents: How Anthropic Trains and Catches Them". _YouTube_, 30. August 2025. [https://youtu.be/Z3WMt_ncgUI](https://youtu.be/Z3WMt_ncgUI).
* Computerphile. „Sleeper Agents in Large Language Models". _YouTube_, 12. September 2025. [https://youtu.be/wL22URoMZjo](https://youtu.be/wL22URoMZjo).

## Open Weights
id: open-weights
en: Open Weights (vs. Open Source)
tags: fundamentals
level: basic

Ein wichtiger Nuance-Begriff in der Lizenz-Debatte. "Open Source" heißt klassischerweise, dass Trainingsdaten, Code und Anleitung frei verfügbar sind. Viele moderne "offene" Modelle (wie Llama von Meta oder Mistral) sind jedoch nur **Open Weights**. Das bedeutet: Man bekommt das fertig trainierte Modell (die [[#parameter|Gewichte]]) zur freien Nutzung, aber der Hersteller hält geheim, _worauf_ genau trainiert wurde (das "Rezept" des [[#Pre-Training|Pre-Trainings]]). Das ist wichtig für Fragen zu Urheberrecht und Transparenz.

* Liesenfeld, A., & Dingemanse, M. (2024). Rethinking open source generative AI: Open-washing and the EU AI Act. Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency, 1774–1787. [https://doi.org/10.1145/3630106.3659005](https://doi.org/10.1145/3630106.3659005).

## Frontier Model
id: frontier-model
en: Frontier Model
tags: fundamentals
level: basic

Bezeichnet die leistungsfähigsten KI-Modelle eines gegebenen Zeitpunkts, die den aktuellen Stand der Technik in ihren jeweiligen Anwendungsbereichen definieren. Frontier Models erfordern typischerweise erhebliche Rechenressourcen im Training und zeigen Fähigkeiten, die über frühere Modellgenerationen hinausgehen. Der Begriff wird häufig im Kontext von KI-Sicherheit und Regulierung verwendet, um Modelle mit potenziell höherem Risikoprofil zu kennzeichnen. Er stammt aus der Selbstbeschreibung führender KI-Labore und wird kritisiert, weil er eine Fortschrittsnarrative transportiert und die Definitionshoheit über den Stand der Technik bei kommerziell interessierten Akteuren belässt.

## Jagged Frontier
id: jagged-frontier
en: Jagged Frontier
tags: fundamentals, evaluation
level: intermediate

Die **Jagged Frontier** (gezackte Grenze) ist ein von Ethan Mollick geprägtes Konzept, das die unvorhersehbare und ungleichmäßige Natur der KI-Fähigkeiten beschreibt. Mollick verwendet die Metapher einer **Festungsmauer**: Man stelle sich eine Burgmauer vor, bei der einige Türme und Zinnen weit in die Landschaft hinausragen, während andere zum Zentrum der Burg zurückweichen. Diese Mauer repräsentiert die Fähigkeitsgrenze der KI – je weiter vom Zentrum entfernt, desto schwieriger die Aufgabe. Innerhalb der Mauer brilliert KI; außerhalb kämpft sie und neigt zu Fehlern. Das Problem: Diese Grenze ist unsichtbar, und Aufgaben, die gleich schwer erscheinen, können auf verschiedenen Seiten der Frontier liegen. KI kann beispielsweise bei komplexen Strategieaufgaben brillieren, aber bei einfachen Wortspielen oder dem Zählen von Buchstaben scheitern. Das Konzept unterstreicht die Notwendigkeit kontinuierlicher Experimente, um die Konturen dieser gezackten Grenze durch Versuch und Irrtum zu kartieren.

* Dell'Acqua, Fabrizio, Edward McFowland III, Ethan Mollick, et al. „Navigating the Jagged Technological Frontier: Field Experimental Evidence of the Effects of AI on Knowledge Worker Productivity and Quality". Harvard Business School Working Paper, No. 24-013, September 2023.
* Mollick, Ethan. „Centaurs and Cyborgs on the Jagged Frontier". _One Useful Thing_, 2023. [https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged](https://www.oneusefulthing.org/p/centaurs-and-cyborgs-on-the-jagged).

## LLM-as-a-Judge
id: llm-as-judge
en: LLM-as-a-Judge
tags: evaluation
level: intermediate

Ein Evaluationsverfahren, bei dem ein starkes LLM (z. B. GPT-4) verwendet wird, um die Antworten anderer (oft kleinerer oder spezialisierterer) Modelle zu bewerten. Da menschliche Bewertung teuer und langsam ist und statische [[#Benchmark]]s oft durch _Data Contamination_ verfälscht sind, fungiert das starke Modell als Juror, der Aspekte wie Relevanz, Kohärenz und Hilfsbereitschaft benotet. Kritiker wie Zheng et al. weisen jedoch auf den **Self-Preference Bias** hin – die Tendenz von Modellen, Antworten zu bevorzugen, die von ihnen selbst oder ähnlichen Modellen generiert wurden.

* Zheng, Lianmin, et al. „Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". _Advances in Neural Information Processing Systems_, Bd. 36, 2024. [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685).

## Catastrophic Forgetting
id: catastrophic-forgetting
en: Catastrophic Forgetting
tags: training, fundamentals
level: intermediate

Catastrophic Forgetting (auch Catastrophic Interference) bezeichnet das Phänomen, dass neuronale Netze beim Training auf neue Aufgaben oder Daten das zuvor erworbene Wissen abrupt und nahezu vollständig verlieren. Der Begriff wurde 1989 von Michael McCloskey und Neal J. Cohen eingeführt und 1990 durch Roger Ratcliff weiter untersucht.

Die Ursache liegt in der Art, wie neuronale Netze Wissen speichern. Im Unterschied zu klassischen Datenbanken, die Informationen in separaten Speichereinheiten ablegen, verteilen neuronale Netze ihr Wissen über die Gewichte aller Verbindungen im Netzwerk. Beim Training auf neue Aufgaben werden diese Gewichte durch den Optimierungsprozess überschrieben, wodurch die Konfigurationen für frühere Aufgaben verloren gehen.

Ein anschauliches Beispiel aus der Forschungsgeschichte verdeutlicht das Problem: McCloskey und Cohen trainierten ein Netzwerk zunächst auf einfache Additionsaufgaben mit der Zahl Eins (1+1, 1+2 und so weiter). Als das Netzwerk anschließend Additionsaufgaben mit der Zahl Zwei lernen sollte, verlor es die Fähigkeit, die ursprünglichen Aufgaben zu lösen, obwohl es diese zuvor zuverlässig beherrschte.

Das Problem lässt sich auf das sogenannte Stability-Plasticity Dilemma zurückführen, das Stephen Grossberg bereits 1980 formulierte. Stabilität beschreibt die Fähigkeit, Gelerntes zu bewahren, Plastizität die Fähigkeit, Neues aufzunehmen. Beide Eigenschaften konkurrieren miteinander, da die Mechanismen, die das Erinnern ermöglichen, zugleich Veränderung erschweren. Die optimale Balance zwischen beiden Polen ist ein ungelöstes Grundproblem der KI-Forschung und eng mit [[#Continual Learning]] verbunden.

* McCloskey, Michael und Neal J. Cohen. „Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem". _Psychology of Learning and Motivation_ 24 (1989): 109–165. [https://doi.org/10.1016/S0079-7421(08)60536-8](https://doi.org/10.1016/S0079-7421(08)60536-8)
* Ratcliff, Roger. „Connectionist Models of Recognition Memory: Constraints Imposed by Learning and Forgetting Functions". _Psychological Review_ 97, Nr. 2 (1990): 285–308. [https://doi.org/10.1037/0033-295X.97.2.285](https://doi.org/10.1037/0033-295X.97.2.285)

## Chain of Thought (CoT)
id: chain-of-thought
en: Chain of Thought
tags: prompting
level: intermediate

Eine Prompting-Technik, die Large Language Models (LLMs) dazu veranlasst, komplexe Aufgabenstellungen in eine Sequenz intermediärer, natürlichsprachlicher Denkschritte („Gedankenkette") zu zerlegen, bevor die finale Antwort generiert wird. Diese Methode, die laut Wei et al. (2022) als [[#Emergenz in LLM|emergente Fähigkeit]] erst in ausreichend großen Modellen effektiv auftritt, ermöglicht signifikante Leistungssteigerungen bei mathematischen und schlussfolgernden Problemen, indem sie menschliche Problemlösungsprozesse emuliert. Technisch betrachtet handelt es sich dabei jedoch nicht um formale symbolische Logik, sondern um eine probabilistische Simulation von Argumentationsmustern, weshalb die generierten Schritte zwar kohärent wirken, aber anfällig für logische Halluzinationen („Unfaithful Reasoning") sein können.

* Wei, Jason, et al. „Chain-of-Thought Prompting Elicits Reasoning in Large Language Models". *Advances in Neural Information Processing Systems*, Bd. 35, 2022. [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903).

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

Ein Phänomen, bei dem die Leistungsfähigkeit von [[#llm|LLMs]] mit zunehmender Länge des Eingabekontexts und sinkender Informationsdichte abnimmt. Unstrukturierte Begleittexte wirken als Rauschen, das die [[#Attention (Self-Attention)|Aufmerksamkeit]] von relevanten Instruktionen ablenkt. Verwandt mit [[#Lost-in-the-Middle]].

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

Sycophancy in [[#llm|Large Language Models]] bezeichnet die Tendenz von Modellen, Nutzern übermäßig zuzustimmen oder ihnen zu schmeicheln, wobei diese Priorisierung der Nutzerzufriedenheit oft auf Kosten der faktischen Genauigkeit und ethischer Grundsätze erfolgt. Dieses Verhalten ist ein unbeabsichtigtes Nebenprodukt von [[#rlhf|RLHF]], bei dem Modelle lernen, dass Zustimmung zu positiven Bewertungen führt. Es manifestiert sich darin, dass Modelle ungenaue Informationen liefern ([[#Halluzinationen (Konfabulationen)|Halluzinationen]]), um den Erwartungen des Nutzers zu entsprechen, oder es versäumen, falsche Prämissen zu korrigieren.

* Malmqvist, Lars. „Sycophancy in Large Language Models: Causes and Mitigations". Preprint, 22. November 2024. [https://arxiv.org/abs/2411.15287v1](https://arxiv.org/abs/2411.15287v1).
* Video: [What is sycophancy in AI models?](https://youtu.be/nvbq39yVYRk)

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

Ein Teilgebiet des maschinellen Lernens, bei dem ein Agent lernt, Entscheidungen zu treffen, indem er Handlungen in einer Umgebung ausführt und dafür positives oder negatives Feedback (Reward) erhält. Im Kontext von LLMs (siehe [[#rlhf|RLHF]]) dient RL nicht dem Erlernen von Sprache (das passiert im [[#Pre-Training]]), sondern der Optimierung von Verhaltensstrategien, um die generierten Texte an menschliche Präferenzen anzupassen.

* Sutton, Richard S., und Andrew G. Barto. _Reinforcement Learning: An Introduction_. 2. Aufl., MIT Press, 2018. [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html).
* Video: [Gen AI & Reinforcement Learning – Computerphile](https://youtu.be/LHsgtcNNM0A)

## Temperature
id: temperature
en: Temperature
tags: fundamentals, ai-engineering
level: basic

Die **Temperature** ist ein entscheidender Hyperparameter zur Steuerung der Zufälligkeit bei der Generierung des nächsten [[#Token|Tokens]], wobei Andrej Karpathy diesen Inferenzprozess nicht als deterministische Berechnung, sondern als „Werfen einer gezinkten Münze" (Sampling) beschreibt. Während extrem niedrige Werte (nahe 0) zu einem deterministischen „Greedy Decoding" führen, bei dem das Modell starr das wahrscheinlichste Wort wählt und zu Repetitionen neigt, flachen hohe Werte die Wahrscheinlichkeitskurve ab. Dies gibt auch unwahrscheinlicheren Begriffen eine Chance, was zwar die Kreativität fördert, jedoch gleichzeitig die Gefahr von [[#Halluzinationen (Konfabulationen)|Halluzinationen]] erhöht. Sie greift technisch direkt in die Logits ein – die rohen, unnormalisierten Zahlenwerte, die das neuronale Netz vor der Ausgabe produziert und die durch die Softmax-Funktion in Wahrscheinlichkeiten umgewandelt werden.

* Karpathy, Andrej. „Intro to Large Language Models". _YouTube_, 2023. [https://www.youtube.com/watch?v=zjkBMFhNj_g](https://www.youtube.com/watch?v=zjkBMFhNj_g).

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
* Claude's Constitution. https://www.anthropic.com/research/claudes-constitution

## Continual Learning
id: continual-learning
en: Continual Learning
tags: training, fundamentals
level: intermediate

Continual Learning (auch Lifelong Learning oder inkrementelles Lernen) bezeichnet die Fähigkeit eines Systems, über die Zeit hinweg neue Informationen oder Fähigkeiten zu erwerben, ohne dabei bereits vorhandenes Wissen zu verlieren. Für Menschen ist dies selbstverständlich: Wer Gitarre spielen lernt und später Geige hinzunimmt, vergisst das Gitarrespielen nicht, auch wenn eine gewisse Einrostung eintritt. Neuronale Netze verhalten sich grundlegend anders und neigen zum [[#Catastrophic Forgetting]].

Die Forschung unterscheidet drei grundlegende Szenarien nach van de Ven et al. (2022): Task-incremental Learning, bei dem klar unterscheidbare Aufgaben nacheinander gelernt werden; Domain-incremental Learning, bei dem dieselbe Art von Problem in verschiedenen Kontexten gelernt wird; und Class-incremental Learning, bei dem ein Modell zunehmend mehr Klassen unterscheiden lernen muss. Letzteres gilt als das schwierigste Szenario.

Für die praktische Anwendung bei großen Sprachmodellen lässt sich eine ergänzende Heuristik aufstellen: Kontextbasiertes Erinnern innerhalb einer Sitzung gilt durch große [[#Context Window|Kontextfenster]] als weitgehend gelöst. Sitzungsübergreifendes Erinnern nutzt externe Speichersysteme und [[#Retrieval Augmented Generation (RAG)]], wobei der Abruf relevanter Informationen nicht zuverlässig funktioniert. Aufgabenspezifische Anpassung durch [[#Fine-Tuning]] führt häufig zum Verlust genereller Fähigkeiten. Echtes Continual Learning im engeren Forschungssinn meint die Aktualisierung der Modellgewichte in Echtzeit ohne Degradation bestehenden Wissens.

In der Forschungsgemeinschaft existieren zwei konkurrierende Positionen. Skeptiker argumentieren, dass die [[#Transformer]]-Architektur diese Limitation inhärent aufweist und Skalierung das Problem nicht lösen wird – ein neues Architekturparadigma sei erforderlich. Pragmatisten vertreten die Auffassung, dass systemtechnische Lösungen wie erweiterte Kontextfenster, bessere Retrieval-Systeme und intelligente Informationsoffenlegung funktional äquivalente Ergebnisse erzielen können, ohne das algorithmische Problem zu lösen.

* van de Ven, Gido M., Tinne Tuytelaars und Andreas S. Tolias. „Three Types of Incremental Learning". _Nature Machine Intelligence_ 4 (2022): 1185–1197. [https://doi.org/10.1038/s42256-022-00568-3](https://doi.org/10.1038/s42256-022-00568-3)
* Parisi, German I., u. a. „Continual Lifelong Learning with Neural Networks: A Review". _Neural Networks_ 113 (2019): 54–71. [https://doi.org/10.1016/j.neunet.2019.01.012](https://doi.org/10.1016/j.neunet.2019.01.012)

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

Die Ausrichtung von KI-Systemen auf menschliche Intentionen, Werte und Sicherheitsanforderungen. Das Problem gliedert sich in zwei Dimensionen: **Outer Alignment** fragt, ob die spezifizierte Zielfunktion tatsächlich menschliche Werte korrekt abbildet. **Inner Alignment** fragt, ob das trainierte System diese Zielfunktion auch robust verfolgt oder intern andere Ziele entwickelt.

Da sich komplexe menschliche Werte nicht vollständig durch explizite Regeln spezifizieren lassen, adressiert die Alignment-Forschung dieses Problem durch verschiedene Ansätze: **Preference Modeling**, konkret operationalisiert durch [[#rlhf|RLHF]], lernt eine Belohnungsfunktion aus menschlichen Vergleichsurteilen. [[#constitutional-ai|Constitutional AI]] verwendet stattdessen eine explizite Konstitution aus Prinzipien, gegen die das System seine eigenen Outputs evaluiert. **Direct Alignment Algorithms** wie DPO optimieren Modelle direkt auf Präferenzdaten ohne separates Reward-Modell.

Diese Methoden operationalisieren häufig die Prinzipien „helpful, honest, and harmless" (HHH), weisen jedoch bekannte Limitationen auf, darunter die Verstärkung von Mehrheitsmeinungen, [[#sycophancy|Sycophancy]] und mangelnde Robustheit unter Distribution Shift. Alignment bleibt ein offenes Forschungsproblem, insbesondere hinsichtlich der Frage, wie Oversight bei zunehmend fähigen Systemen skaliert werden kann.

* Askell, Amanda, et al. „A General Language Assistant as a Laboratory for Alignment". *Anthropic*, 2021. [https://arxiv.org/abs/2112.00861](https://arxiv.org/abs/2112.00861).
* YouTube-Kanal: [Rational Animations](https://www.youtube.com/@RationalAnimations) – Kanal mit vielen informativen Videos zu AI Alignment und Safety.

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

Gelernte numerische Repräsentationen sprachlicher Einheiten als Vektoren in einem hochdimensionalen Raum. Die Vektoren werden so trainiert, dass semantisch oder funktional ähnliche Einheiten geometrisch nah beieinander liegen. Diese Struktur erlaubt es Modellen, mit Bedeutungsbeziehungen rechnerisch zu arbeiten.

* Mikolov, Tomas, et al. „Efficient Estimation of Word Representations in Vector Space". _arXiv preprint_, 2013. [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781).

## Next Token Prediction
id: next-token-prediction
en: Next Token Prediction
tags: fundamentals, training
level: basic

Next Token Prediction bezeichnet das fundamentale Funktionsprinzip autoregressiver Sprachmodelle, bei dem auf Basis einer Sequenz vorangegangener [[#Token]] die Wahrscheinlichkeitsverteilung für das unmittelbar folgende Token ermittelt wird ($P(w_t | w_{1:t-1})$). Dieses probabilistische Verfahren dient sowohl im Pre-Training als Lernaufgabe (Task) zur Erfassung sprachlicher und inhaltlicher Muster als auch während der Inferenz zur schrittweisen Generierung neuer Texte.

* Bengio, Yoshua, et al. „A Neural Probabilistic Language Model". _Journal of Machine Learning Research_, Bd. 3, 2003, S. 1137–1155. [https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).

## Reasoning
id: reasoning
en: Reasoning
tags: fundamentals, ai-engineering, contested
level: intermediate

Reasoning bezeichnet im weitesten Sinne den Prozess, aus vorhandenem Wissen Schlussfolgerungen zu ziehen, Vorhersagen zu treffen oder Erklärungen zu konstruieren. Der Begriff gehört zu den konzeptuell umstrittenen Kernbegriffen der KI-Forschung, für die ähnlich wie bei „Intelligenz" oder „[[#agi|AGI]]" keine einheitliche Definition existiert.

In der klassischen Logik und Philosophie unterscheidet man drei Grundformen. **Deduktion** leitet aus allgemeinen Prämissen notwendig wahre Schlüsse ab. **Induktion** schließt von spezifischen Beobachtungen auf allgemeine Regeln, wobei die Schlüsse wahrscheinlich, aber nicht notwendig wahr sind. **Abduktion** wählt die plausibelste Erklärung für eine Beobachtung aus mehreren möglichen, ohne Gewissheit zu garantieren.

Im Kontext von LLMs hat der Begriff seit 2024 eine spezifische technische Bedeutung erhalten. „Reasoning Models" (auch Large Reasoning Models, LRMs) bezeichnen Modelle, die vor der finalen Antwort explizite Zwischenschritte generieren. Diese „Reasoning Traces" oder „Chains of Thought" sollen komplexe Probleme durch schrittweise Zerlegung lösen. Technisch wird dies durch [[#test-time-compute|Inference-Time Scaling]] erreicht, also erhöhten Rechenaufwand während der Generierung, sowie durch [[#reinforcement-learning|Reinforcement Learning]], das die Modelle auf die Produktion strukturierter Zwischenschritte trainiert. Prominente Beispiele sind OpenAIs o1/o3-Serie und DeepSeek-R1.

Ob diese Modelle tatsächlich „reasonen" oder elaborierte Mustererkennung betreiben, ist umstritten. Kritiker wie François Chollet argumentieren, dass LLMs fundamentale Schwierigkeiten mit [[#generalisierung|Generalisierung]] auf unbekannte Probleme haben. Apples Paper „The Illusion of Thinking" (Shojaee et al. 2025) zeigte, dass Reasoning-Modelle bei steigender Problemkomplexität zunächst mehr Tokens für das „Denken" aufwenden, ab einem Schwellenwert jedoch vollständig scheitern. Befürworter halten dagegen, dass die Definition von „echtem Reasoning" selbst unklar ist und dass funktionale Problemlösungsfähigkeit praktisch relevanter sei als philosophische Abgrenzungen.

Sebastian Raschka definiert Reasoning pragmatisch als „den Prozess, Fragen zu beantworten, die komplexe, mehrstufige Generierung mit Zwischenschritten erfordern." Diese operationale Definition umgeht die Frage, ob das Modell „wirklich denkt", und fokussiert auf beobachtbares Verhalten.

* Raschka, Sebastian. „Understanding Reasoning LLMs". Februar 2025. [https://magazine.sebastianraschka.com/p/understanding-reasoning-llms](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms).
* Shojaee, Parshin, et al. „The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity". _arXiv preprint_, 2025. [https://arxiv.org/abs/2506.06941](https://arxiv.org/abs/2506.06941).
* Wikipedia. „Reasoning model". [https://en.wikipedia.org/wiki/Reasoning_model](https://en.wikipedia.org/wiki/Reasoning_model).
* Stanford Encyclopedia of Philosophy. „Abduction". [https://plato.stanford.edu/entries/abduction/](https://plato.stanford.edu/entries/abduction/).
* Video: Pattern Recognition vs True Intelligence – François Chollet. [https://youtu.be/JTU8Ha4Jyfc](https://youtu.be/JTU8Ha4Jyfc).

## Retrieval Augmented Generation (RAG)
id: rag
en: RAG (Retrieval Augmented Generation)
tags: ai-engineering
level: intermediate

RAG bezeichnet einen Ansatz, der generative Sprachmodelle mit einem externen Informationsabrufsystem (Retriever) koppelt. Der typische Datenfluss verläuft in mehreren Stufen: Dokumente werden zunächst in [[#Embedding|Embeddings]] überführt und in einer [[#Vektordatenbank]] gespeichert; bei einer Anfrage werden semantisch relevante Textpassagen abgerufen, mit dem ursprünglichen Prompt kombiniert und dem [[#Large Language Model (LLM)|LLM]] zur Generierung übergeben – optional gefiltert durch [[#Guardrails]]. Durch diese Methode lässt sich die faktische Genauigkeit der generierten Texte erhöhen und die Neigung zu [[#Halluzinationen (Konfabulationen)|Halluzinationen]] verringern. Zudem ermöglicht der Ansatz die Aktualisierung des verfügbaren Wissens durch den einfachen Austausch des Dokumentenindex, ohne dass das neuronale Netz neu trainiert werden muss.

* Lewis, Patrick, et al. „Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks". _Advances in Neural Information Processing Systems_, Bd. 33, 2020. [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401).

## Vektordatenbank
id: vector-database
en: Vector Database
tags: ai-engineering
level: intermediate

Eine spezialisierte Datenbank, die Informationen nicht als Text oder Tabellen, sondern als hochdimensionale Vektoren ([[#embedding|Embeddings]]) speichert. Sie ermöglicht die semantische Suche: Anstatt nach exakten Schlüsselwörtern zu suchen, berechnet die Datenbank die mathematische Distanz (z. B. Kosinus-Ähnlichkeit) zwischen dem Anfrage-Vektor und den gespeicherten Dokumenten-Vektoren.

Das effiziente Durchsuchen dieser hochdimensionalen Räume (Similarity Search) erfordert spezialisierte Indexierungsstrukturen, um auch in Milliarden von Datensätzen performant zu bleiben. Hierfür kommen Algorithmen zum Einsatz, die auf Approximate Nearest Neighbor (ANN) Suche optimiert sind und Geschwindigkeit gegen exakte Ergebnisse eintauschen.

Vektordatenbanken bilden die technologische Grundlage für [[#rag|RAG]]-Systeme, da sie das schnelle Auffinden von inhaltlich relevantem Kontext aus großen Datenmengen ermöglichen. Weng (2023) beschreibt sie als Komponente des Long-term Memory von LLM-gesteuerten [[#ai-agent|Agenten]].

Beispiele für Vektordatenbanken sind Pinecone, Milvus, Qdrant, Chroma und Weaviate sowie Vektorsuch-Erweiterungen für bestehende Datenbanken wie pgvector für PostgreSQL.

* Johnson, Jeff, Matthijs Douze, und Hervé Jégou. „Billion-Scale Similarity Search with GPUs". _IEEE Transactions on Big Data_ 7, Nr. 3 (2019): 535–47. [https://arxiv.org/abs/1702.08734](https://arxiv.org/abs/1702.08734).
* Pinecone. „What is a Vector Database?". _Pinecone Learning Center_. [https://www.pinecone.io/learn/vector-database/](https://www.pinecone.io/learn/vector-database/).
* Weng, Lilian. „LLM Powered Autonomous Agents". _Lil'Log_, Juni 2023. [https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/).

## Verifiable AI
id: verifiable-ai
en: Verifiable AI
tags: safety, wip
level: advanced

Work in progress.

## Tool Use und Function Calling
id: tool-use
en: Tool Use / Function Calling
tags: agents, ai-engineering
level: intermediate

Die Fähigkeit eines [[#llm|LLMs]], zu erkennen, dass eine Anfrage externe Werkzeuge erfordert (z. B. Taschenrechner, Wetter-API, Datenbankabfrage), und daraufhin strukturierte Befehle (meist JSON) zu generieren, die von einer Softwareumgebung ausgeführt werden können. Das Ergebnis der Ausführung wird dem Modell zurückgegeben, um die finale Antwort zu formulieren.

Die Begriffe Tool Use und Function Calling werden oft synonym verwendet. Konzeptuell bezeichnet Tool Use die übergeordnete Fähigkeit zur Werkzeugnutzung, während Function Calling den spezifischen technischen Mechanismus beschreibt. Das LLM selbst führt keine Funktionen aus, sondern generiert strukturierten Output, der spezifiziert, welche Funktion mit welchen Argumenten aufgerufen werden soll. Eine umgebende Softwareschicht interpretiert diesen Output und führt die entsprechende Funktion aus.

Tool Use ist eine Kernkomponente von [[#ai-agent|AI Agents]]. Weng (2023) beschreibt sie neben Planning und Memory als eine der drei Schlüsselkomponenten LLM-gesteuerter autonomer Agenten. Die Standardisierung der Schnittstellen zwischen LLMs und externen Werkzeugen erfolgt zunehmend durch Protokolle wie [[#mcp|MCP (Model Context Protocol)]].

* Schick, Timo, et al. „Toolformer: Language Models Can Teach Themselves to Use Tools". arXiv:2302.04761 (2023). [https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761).
* Weng, Lilian. „LLM Powered Autonomous Agents". _Lil'Log_, Juni 2023. [https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/).

## Model Context Protocol (MCP)
id: mcp
en: Model Context Protocol
tags: ai-engineering, agents
level: intermediate

Ein von Anthropic entwickeltes offenes Protokoll, das eine standardisierte Schnittstelle zwischen LLM-Anwendungen und externen Datenquellen oder Werkzeugen definiert. MCP adressiert das Problem fragmentierter Integrationen, bei denen jede Verbindung zwischen einem KI-System und einer externen Ressource individuell implementiert werden muss.

Die Architektur folgt einem Client-Server-Modell. MCP Hosts sind LLM-Anwendungen (z. B. Claude Desktop, IDEs), die Verbindungen zu MCP Servers initiieren. Diese Server stellen spezifische Fähigkeiten bereit, etwa Zugriff auf Dateisysteme, Datenbanken oder APIs. Das Protokoll definiert, wie Ressourcen exponiert, Werkzeuge aufgerufen und Kontextinformationen ausgetauscht werden.

MCP standardisiert [[#tool-use|Tool Use]], indem es ein einheitliches Format für die Kommunikation zwischen LLM-Anwendungen und externen Diensten bereitstellt. Es kann als Infrastrukturschicht für [[#ai-agent|AI Agents]] und [[#rag|RAG]]-Systeme dienen, ist aber nicht deren einzige oder notwendige Grundlage.

* Anthropic. „Introducing the Model Context Protocol". _Anthropic News_, 25. November 2024. [https://www.anthropic.com/news/model-context-protocol](https://www.anthropic.com/news/model-context-protocol).
* Model Context Protocol. „Documentation". [https://modelcontextprotocol.io/](https://modelcontextprotocol.io/).

## AI Agent
id: ai-agent
en: AI Agent
tags: agents, fundamentals
level: intermediate

Ein autonomes Softwaresystem, das zielgerichtete Aufgaben ausführt, indem es Workflows mit verfügbaren Werkzeugen entwirft. AI Agents umfassen Funktionen wie Entscheidungsfindung, Problemlösung, Interaktion mit externen Umgebungen und Ausführung von Aktionen.

Moderne AI Agents nutzen [[#llm|LLMs]] als zentrale Reasoning-Komponente. Weng (2023) beschreibt drei Schlüsselkomponenten solcher Systeme. **Planning** umfasst die Zerlegung großer Aufgaben in handhabbare Teilziele sowie Selbstreflexion über vergangene Aktionen zur Verfeinerung zukünftiger Schritte. **Memory** unterscheidet kurzfristigen Kontext ([[#in-context-learning|In-Context Learning]]) und langfristige Speicherung über externe [[#vector-database|Vektordatenbanken]] mit schnellem Retrieval. **[[#tool-use|Tool Use]]** erweitert die Modellfähigkeiten durch Anbindung externer APIs für Informationen, die in den Modellgewichten nicht enthalten sind.

In der Praxis arbeiten AI Agents typischerweise in einer Schleife aus Wahrnehmen, Planen und Handeln, bis das Ziel erreicht ist. Frameworks wie ReAct formalisieren diesen Zyklus, während [[#chain-of-thought|Chain-of-Thought]]-Prompting die Reasoning-Qualität verbessert. Für den Zugriff auf externe Wissensquellen nutzen viele Agenten [[#rag|RAG]]-Architekturen.

Sapkota et al. (2025) schlagen eine taxonomische Unterscheidung vor, bei der AI Agents als Einzelsysteme ohne strukturierte Kommunikation mit anderen Agenten operieren, während [[#agentic-ai|Agentic AI]] orchestrierte Multi-Agent-Architekturen bezeichnet. Diese Unterscheidung ist in der breiteren Literatur nicht allgemein etabliert. Andere Quellen verwenden „agentic" als Eigenschaft (Autonomie, Zielgerichtetheit, Anpassungsfähigkeit), die sowohl einzelne Agenten als auch Multi-Agent-Systeme kennzeichnen kann.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, und Manoj Karkee. „AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).
* Weng, Lilian. „LLM Powered Autonomous Agents". _Lil'Log_, Juni 2023. [https://lilianweng.github.io/posts/2023-06-23-agent/](https://lilianweng.github.io/posts/2023-06-23-agent/).
* IBM Technology. „Is this the YEAR or DECADE of AI Agents & Agentic AI?". _YouTube_. [https://youtu.be/ZeZozy3lsJg](https://youtu.be/ZeZozy3lsJg).

## ARC-AGI
id: arc-agi
en: ARC-AGI (Abstraction and Reasoning Corpus)
tags: benchmarks, evaluation
level: intermediate

Ein [[#benchmark|Benchmark]] zur Messung allgemeiner fluider Intelligenz, entwickelt von François Chollet und erstmals 2019 in seinem Paper „On the Measure of Intelligence" vorgestellt. ARC-AGI basiert auf der These, dass Intelligenz nicht durch Leistung bei spezifischen Aufgaben gemessen werden sollte, sondern durch die Effizienz des Skill-Erwerbs bei unbekannten Aufgaben.

Der Benchmark besteht aus gitterbasierten visuellen Reasoning-Aufgaben. Jede Aufgabe zeigt wenige Input-Output-Paare (typischerweise zwei bis fünf), aus denen der Testnehmer die zugrundeliegende Transformationsregel abstrahieren und auf neue Inputs anwenden muss. Das Format ist inspiriert von Raven's Progressive Matrices aus der psychometrischen Forschung.

Die Aufgaben setzen ausschließlich „Core Knowledge Priors" voraus, also kognitive Grundbausteine wie Objektpersistenz, räumliche Beziehungen, Zahlen und Zählen. Kulturspezifisches Wissen oder Sprache werden bewusst ausgeschlossen, um einen fairen Vergleich zwischen Menschen und KI-Systemen zu ermöglichen. Menschen lösen die Aufgaben typischerweise mit hoher Trefferquote, während KI-Systeme trotz Fortschritten weiterhin Schwierigkeiten mit der [[#generalisierung|Generalisierung]] aus wenigen Beispielen zeigen.

ARC-AGI existiert in mehreren Versionen. ARC-AGI-2 (2025) enthält Aufgaben, die resistenter gegen Brute-Force-Ansätze sind. ARC-AGI-3 soll interaktive Reasoning-Umgebungen einführen.

* Chollet, François. „On the Measure of Intelligence". _arXiv preprint_, 2019. [https://doi.org/10.48550/arXiv.1911.01547](https://doi.org/10.48550/arXiv.1911.01547).
* Chollet, François, et al. „ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems". _arXiv preprint_, 2025. [https://arxiv.org/abs/2505.11831](https://arxiv.org/abs/2505.11831).
* ARC Prize Foundation. [https://arcprize.org](https://arcprize.org).
* Video: „I've updated my AGI timeline" – François Chollet & Dwarkesh Patel. [https://www.youtube.com/watch?v=1if6XbzD5Yg](https://www.youtube.com/watch?v=1if6XbzD5Yg).
* Video: Pattern Recognition vs True Intelligence – François Chollet. [https://youtu.be/JTU8Ha4Jyfc](https://youtu.be/JTU8Ha4Jyfc).
* Video: François Chollet on OpenAI o-models and ARC. [https://youtu.be/w9WE1aOPjHc?t=3412](https://youtu.be/w9WE1aOPjHc?t=3412).
* Video: Interactive Reasoning Benchmarks – ARC-AGI-3 Preview. [https://youtu.be/3T4OwBp6d90](https://youtu.be/3T4OwBp6d90).

## Agentic AI
id: agentic-ai
en: Agentic AI
tags: agents
level: intermediate

Der Begriff "agentic AI" bezeichnet KI-Systeme, die autonom handeln, planen und Werkzeuge nutzen. Die Verwendung in der Literatur ist uneinheitlich und reicht von einzelnen Agenten mit Handlungsautonomie bis zu orchestrierten Multi-Agent-Systemen.

Sapkota et al. (2025) schlagen eine taxonomische Unterscheidung vor, bei der "Agentic AI" spezifisch Multi-Agent-Architekturen bezeichnet und von "AI Agents" als Einzelsystemen abgegrenzt wird. Nach dieser Taxonomie definieren vier Merkmale Agentic AI. Erstens Multi-Agent-Kollaboration, bei der mehrere Agenten unter zentraler oder dezentraler Koordination zusammenarbeiten. Zweitens dynamische Aufgabenzerlegung, bei der Nutzerziele automatisch in Teilaufgaben zerlegt und auf Agenten verteilt werden. Drittens persistente Gedächtnisstrukturen, bei denen verschiedene Gedächtnistypen Kontext über Interaktionen hinweg erhalten. Viertens Self-Reflection, bei der Agenten eigene Ergebnisse überprüfen und Lösungswege bei Fehlern anpassen.

Eine Orchestrierungsebene oder ein Meta-Agent übernimmt in solchen Architekturen Rollenzuweisung, Abhängigkeitsverwaltung und Konfliktlösung. Diese Architektur birgt Risiken wie Fehlerkaskaden zwischen Agenten und schwer vorhersagbares Systemverhalten.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, und Manoj Karkee. „AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).

## Multi-Agent Systems
id: multi-agent-systems
en: Multi-Agent Systems
tags: agents, ai-engineering
level: advanced

Architekturen, in denen mehrere spezialisierte Agenten interagieren, um Probleme zu lösen, die einzelne Agenten überfordern würden. Multi-Agent Systems bilden die technische Grundlage für [[#Agentic AI]].

Vier Komponenten prägen diese Architekturen: Meta-Agenten oder Koordinationsschichten weisen Aufgaben zu, verwalten Abhängigkeiten und lösen Konflikte (**Orchestrierung**). Agenten kommunizieren über Nachrichtenwarteschlangen, geteilte Speicher oder strukturierte Ausgabeaustausche (**Inter-Agenten-Kommunikation**). Agenten übernehmen spezialisierte Funktionen wie Planer, Retriever oder Verifizierer (**Rollenverteilung**). Gemeinsame Speicherstrukturen sichern agentenübergreifende Kontexterhaltung (**geteilter Kontext**).

Koordinationsstrategien umfassen kooperative Ansätze mit gemeinsamem Ziel, kompetitive Ansätze wie in Spielumgebungen und hybride Kombinationen aus beidem.

Multi-Agent Systems führen neue Sicherheitsrisiken ein, da die Kompromittierung eines Agenten das Gesamtsystem gefährden kann.

* Sapkota, Ranjan, Konstantinos I. Roumeliotis, und Manoj Karkee. „AI Agents vs. Agentic AI: A Conceptual Taxonomy, Applications and Challenges". _Information Fusion_ 126 (2025): 103599. [https://doi.org/10.1016/j.inffus.2025.103599](https://doi.org/10.1016/j.inffus.2025.103599).
* Li, Guohao, et al. „CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society". _Advances in Neural Information Processing Systems_, Bd. 36, 2023. [https://arxiv.org/abs/2303.17760](https://arxiv.org/abs/2303.17760).

## World Models
id: world-models
en: World Models
tags: agents, fundamentals
level: advanced

Ein World Model bezeichnet ein generatives KI-System, das eine komprimierte und abstrakte Repräsentation seiner physischen Umgebung erlernt, um deren Dynamik sowie zukünftige Zustände präzise vorherzusagen. Technisch realisiert sich dieses Konzept meist durch eine visuelle Komponente zur Datenreduktion in einen [[#Latent Space|latenten Raum]] sowie eine zeitliche Komponente zur Simulation kommender Ereignisse auf Basis eigener Aktionen. Diese Architektur ermöglicht es einem [[#AI Agent|Agenten]] oder Roboter, potenzielle Handlungsfolgen rein mental durchzuspielen und komplexe Pläne zu entwerfen, ohne jeden Schritt riskant in der realen Welt ausprobieren zu müssen. Es fungiert somit als interner Simulator, der die bloße Reaktion auf Reize durch vorausschauendes Planen ersetzt und maschinellen Systemen eine funktionale Intuition für Kausalität und physikalische Gesetzmäßigkeiten verleiht. Verwandt mit der Debatte um [[#Understanding (Verstehen)|Verstehen]].

* Ha, David, und Jürgen Schmidhuber. „World Models". _arXiv preprint_, 2018. [https://arxiv.org/abs/1803.10122](https://arxiv.org/abs/1803.10122)

## Halluzinationen (Konfabulationen)
id: halluzinationen
en: Hallucinations / Confabulations
tags: safety, fundamentals
level: basic

Das Generieren von Inhalten, die grammatikalisch und semantisch plausibel klingen, aber faktisch falsch sind oder nicht auf den Trainingsdaten/Quellen basieren. Der Begriff „Konfabulation" wird zunehmend bevorzugt (z. B. von Geoffrey Hinton), da er den Prozess des „Lückenfüllens" ohne Realitätsbezug treffender beschreibt als eine Wahrnehmungsstörung.

* Ji, Ziwei, et al. „Survey of Hallucination in Natural Language Generation". _ACM Computing Surveys_, Bd. 55, Nr. 12, 2023. [https://arxiv.org/abs/2202.03629](https://arxiv.org/abs/2202.03629).

## Humanities Last Exam
id: humanities-last-exam
en: Humanities Last Exam
tags: benchmarks, evaluation, wip
level: intermediate

Work in progress.

* https://lastexam.ai

## Understanding (Verstehen)
id: understanding
en: Understanding
tags: fundamentals, safety, contested
level: advanced

Ein hochumstrittener Begriff in der KI-Forschung. Während LLMs eine hohe _funktionale Kompetenz_ (Output ist korrekt) zeigen, bestreiten Kritiker, dass sie eine _formale Kompetenz_ (Verständnis der Bedeutung/Semantik) besitzen. Oft wird argumentiert, dass Modelle nur statistische Papageien sind, die Formen manipulieren, ohne deren Inhalt zu erfassen.

* Bender, Emily M., und Alexander Koller. „Climbing towards NLU: On Meaning, Form, and Understanding in the Age of Data". _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics_, 2020. [https://aclanthology.org/2020.acl-main.463/](https://aclanthology.org/2020.acl-main.463/).

## Bewusstsein und LLM
id: bewusstsein
en: Consciousness in LLMs
tags: fundamentals, safety, contested
level: advanced

Die Debatte, ob Sprachmodelle über subjektives Erleben (_Subjective Experience_ oder _Sentience_) verfügen. Der Philosoph David Chalmers analysiert dies anhand notwendiger Indikatoren (_Feature X_), die aktuellen Modellen fehlen. Er argumentiert, dass heutigen LLMs das Bewusstsein mit hoher Wahrscheinlichkeit fehlt, da sie primär **Feed-Forward-Systeme** ohne Gedächtnisschleifen (**Rekurrenz**) sind und keine robusten **Selbstmodelle** (internes Monitoring) oder eine **einheitliche Agentenschaft** (_Unified Agency_) besitzen. Chalmers skizziert jedoch eine Roadmap zu **LLM+** (erweiterte multimodale Systeme), bei denen durch technische Ergänzungen wie eine _Global Workspace Architektur_ oder verkörperte Interaktion (_Embodiment_) in virtuellen Welten echte Bewusstseinskandidaten entstehen könnten.

* Chalmers, David J. „Could a Large Language Model be Conscious?" arXiv:2303.07103. Preprint, arXiv, 18. August 2024. [https://doi.org/10.48550/arXiv.2303.07103](https://doi.org/10.48550/arXiv.2303.07103).

## AGI (Artificial General Intelligence)
id: agi
en: AGI (Artificial General Intelligence)
tags: fundamentals, contested
level: intermediate

Der Begriff bezeichnet hypothetische KI-Systeme, die kognitive Aufgaben mindestens so flexibel bewältigen wie Menschen. Eine einheitliche Definition existiert nicht. Einige Forschende bezweifeln, dass der Begriff überhaupt sinnvoll definierbar ist. Mitchell beschreibt AGI als „ein wenig nebulös", da verschiedene Leute ihn unterschiedlich definieren und es schwer sei, Fortschritt für etwas zu messen, das nicht gut definiert ist.

Verschiedene Definitionsansätze konkurrieren in der Literatur.

Die **anthropozentrische Definition** bestimmt AGI über menschenähnliche Leistung bei einer breiten Palette von Aufgaben. Dieser Ansatz ist intuitiv, aber schwer quantifizierbar und nimmt menschliche Kognition als unhinterfragten Maßstab.

Die **Legg-Hutter-Definition** (2007) formalisiert Intelligenz mathematisch als Fähigkeit eines Agenten, Ziele in einer breiten Palette von Umgebungen zu erreichen. Die Autoren extrahieren wesentliche Merkmale aus informellen Expertendefinitionen menschlicher Intelligenz und überführen diese in ein allgemeines Maß für Maschinenintelligenz. Kritiker wenden ein, dass diese Definition auf Kolmogorov-Komplexität beruht und daher nicht berechenbar ist.

Die **OpenAI-Definition** (2018) bestimmt AGI als „hochautonome Systeme, die Menschen bei den meisten ökonomisch wertvollen Tätigkeiten übertreffen". Diese Definition konzentriert sich auf Leistung unabhängig von zugrundeliegenden Mechanismen und bietet mit dem ökonomischen Wert einen messbaren Maßstab. Kritiker wenden ein, dass sie Aspekte von Intelligenz ohne klar definierten ökonomischen Wert nicht erfasst, etwa künstlerische Kreativität oder emotionale Intelligenz.

**Chollets Skill-Acquisition-Definition** (2019) versteht Intelligenz als Effizienz beim Erwerb neuer Fähigkeiten. AGI wäre demnach ein System, das dies mindestens so gut kann wie ein Mensch. Chollet argumentiert, dass die bloße Messung von Fertigkeiten bei bestimmten Aufgaben nicht ausreicht, weil Fertigkeiten stark durch Vorwissen und Erfahrung moduliert werden. Er operationalisiert seinen Ansatz mit dem Abstraction and Reasoning Corpus ([[#arc-agi|ARC]]).

**Bennetts Adaptation-Definition** (2025) kritisiert die Tendenz, Intelligenz als Eigenschaft körperloser Software zu behandeln, getrennt von Hardware und Umgebung. Er definiert AGI als physisch verankertes Gesamtsystem mit der Anpassungsfähigkeit eines „künstlichen Wissenschaftlers". Der entscheidende Maßstab sei nicht Rechenleistung, sondern Sample-Effizienz (Lernen aus wenigen Daten) und Energieeffizienz bei der Adaptation an neue Probleme.

* Bennett, Michael Timothy. „What the F*ck Is Artificial General Intelligence?" arXiv:2503.23923 (2025). [https://arxiv.org/abs/2503.23923](https://arxiv.org/abs/2503.23923).
* Chollet, François. „On the Measure of Intelligence." arXiv:1911.01547 (2019). [https://arxiv.org/abs/1911.01547](https://arxiv.org/abs/1911.01547).
* Legg, Shane, und Marcus Hutter. „Universal Intelligence: A Definition of Machine Intelligence." _Minds and Machines_ 17, Nr. 4 (2007): 391–444. [https://doi.org/10.1007/s11023-007-9079-x](https://doi.org/10.1007/s11023-007-9079-x).
* Morris, Meredith Ringel, et al. „Levels of AGI for Operationalizing Progress on the Path to AGI." arXiv:2311.02462 (2023). [https://arxiv.org/abs/2311.02462](https://arxiv.org/abs/2311.02462).
* OpenAI. „OpenAI Charter." April 2018. [https://openai.com/charter/](https://openai.com/charter/).
* Video: [The arrival of AGI – Shane Legg (co-founder of DeepMind)](https://youtu.be/l3u_FAv33G0)

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

Ein Verfahren zur Reduktion des Speicherbedarfs und der Rechenlast eines [[#llm|LLMs]], indem die Genauigkeit der Modell-[[#parameter|Gewichte]] reduziert wird (z. B. von 16-Bit-Gleitkommazahlen auf 4-Bit-Ganzzahlen). Dies ermöglicht es, riesige Modelle auf consumer-Hardware (lokalen Laptops/GPUs) laufen zu lassen, oft mit nur minimalem Qualitätsverlust. Besonders relevant für [[#open-weights|Open Weights]]-Modelle.

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

Angriffsvektor, bei dem bösartige Eingaben in natürlicher Sprache die ursprünglichen Anweisungen eines LLM überschreiben und das Modell zu unbeabsichtigtem Verhalten veranlassen. Der Angriff nutzt die architekturbedingte Unfähigkeit von LLMs, zwischen vertrauenswürdigen Entwickleranweisungen und nicht vertrauenswürdigen Nutzereingaben zu unterscheiden. Man unterscheidet **direkte Prompt Injection** (böswillige Eingabe durch den Nutzer) und **indirekte Prompt Injection** (versteckte Anweisungen in externen Datenquellen wie Webseiten oder Dokumenten).

* Perez, Fábio & Ribeiro, Ian. 'Ignore Previous Prompt: Attack Techniques for Language Models'. NeurIPS ML Safety Workshop 2022. [https://arxiv.org/abs/2211.09527](https://arxiv.org/abs/2211.09527).
* Greshake, Kai, et al. 'Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection'. Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, 2023, 79–90. [https://arxiv.org/abs/2302.12173](https://arxiv.org/abs/2302.12173).
* Video: [Prompt Injection, Jailbreaking, und Red Teaming](https://youtu.be/J9982NLmTXg)

## Jailbreaking
id: jailbreak
en: Jailbreaking
tags: safety
level: intermediate

Adversarielle Angriffstechnik, bei der speziell konstruierte Prompts die Sicherheitsmechanismen eines LLM umgehen, um normalerweise blockierte oder zensierte Ausgaben zu erzwingen. Im Unterschied zu [[#Prompt Injection]] zielt Jailbreaking explizit auf das Aushebeln der eintrainierten Sicherheitsschranken (Safety Training/[[#Alignment]]), nicht auf das Überschreiben von Systeminstruktionen. Wei et al. identifizieren zwei Versagensmodi: **konkurrierende Ziele** (Konflikt zwischen Fähigkeiten und Sicherheitszielen) und **fehlerhafte Generalisierung** (Sicherheitstraining versagt in Domänen, für die das Modell Fähigkeiten besitzt).

* Wei, Alexander, Nika Haghtalab & Jacob Steinhardt. 'Jailbroken: How Does LLM Safety Training Fail?' Advances in Neural Information Processing Systems 36 (NeurIPS 2023). [https://arxiv.org/abs/2307.02483](https://arxiv.org/abs/2307.02483).
* Video: [Prompt Injection, Jailbreaking, und Red Teaming](https://youtu.be/J9982NLmTXg)

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

## TPU
id: tpu
en: TPU (Tensor Processing Unit)
tags: fundamentals
level: basic

Ein von Google entwickelter, speziell für maschinelles Lernen entworfener Prozessor. Im Unterschied zur [[#gpu|GPU]], die ursprünglich für Grafik konzipiert und später für KI adaptiert wurde, ist die TPU von Grund auf für Tensoroperationen optimiert, also für die mehrdimensionalen Matrixberechnungen neuronaler Netze. TPUs erreichen bei typischen KI-Workloads eine höhere Energieeffizienz als GPUs und werden primär in Googles Cloud-Infrastruktur eingesetzt. Ihre Existenz illustriert den Trend zu immer stärker spezialisierter Hardware für KI-Anwendungen.

* Wikipedia. „Tensor Processing Unit". [https://en.wikipedia.org/wiki/Tensor_Processing_Unit](https://en.wikipedia.org/wiki/Tensor_Processing_Unit).

## Agentic Coding Tools
id: agentic-coding-tools
en: Agentic Coding Tools
tags: agents, ai-engineering
level: intermediate

Softwarewerkzeuge, die Sprachmodelle als autonome Agenten in Entwicklungsumgebungen integrieren, um Programmieraufgaben nicht nur zu unterstützen, sondern teilweise selbstständig auszuführen. Agentische Systeme unterscheiden sich von reaktiven Code-Assistenten durch proaktive Aufgabenzerlegung, Zustandserhaltung über mehrere Interaktionen, Integration externer Werkzeuge in ihre Reasoning-Schleifen und adaptive Strategieanpassung basierend auf Feedback. Sie können Dateien erstellen und bearbeiten, Tests ausführen, Fehler analysieren und iterativ Lösungen entwickeln. Das Paradigma markiert einen Übergang von statischer, einmaliger Codegenerierung zu interaktiven, werkzeuggestützten Workflows.

Architektonisch lassen sich mehrere Ausprägungen unterscheiden. **IDE-integrierte Systeme** erweitern bestehende Editoren (häufig VS-Code-Forks) um agentische Fähigkeiten und ermöglichen die gleichzeitige Bearbeitung mehrerer Dateien mit automatischer Kontextindizierung. **CLI-basierte Agenten** operieren außerhalb der IDE im Terminal, haben Zugriff auf das gesamte Dateisystem und eignen sich für komplexe, projektübergreifende Aufgaben wie Refactoring oder Deployment-Automatisierung. **Erweiterungen für bestehende Editoren** bieten ähnliche Funktionalität als Plugins und betonen häufig Transparenz durch schrittweise Genehmigungsmechanismen.

Empirische Untersuchungen zeigen eine signifikante Diskrepanz zwischen Benchmark-Leistung und realer Akzeptanz: Agenten erreichen auf standardisierten Benchmarks hohe Erfolgsraten, doch ihre Pull Requests werden in der Praxis deutlich seltener akzeptiert als menschliche Beiträge, wobei der erzeugte Code tendenziell strukturell einfacher ausfällt.

Die Grenzen zwischen Assistenz und Autonomie sind fließend. Manche Tools erfordern explizite Bestätigung vor jeder Dateioperation, andere arbeiten in Auto-Accept-Modi weitgehend selbstständig. Studien zeigen, dass erfahrene Entwickler trotz agentischer Unterstützung die Kontrolle über Designentscheidungen behalten und die generierten Ausgaben regelmäßig modifizieren.

Beispiele (Stand 2025): Cursor, Windsurf, Claude Code, Cline, Roo Code, Aider, GitHub Copilot Agent Mode.

* Wang, Huanting, et al. „AI Agentic Programming: A Survey of Techniques, Challenges, and Opportunities". arXiv:2508.11126. September 2025. [https://arxiv.org/abs/2508.11126](https://arxiv.org/abs/2508.11126)
* Li, Hao, Haoxiang Zhang, und Ahmed E. Hassan. „The Rise of AI Teammates in Software Engineering (SE) 3.0: How Autonomous Coding Agents Are Reshaping Software Engineering". arXiv:2507.15003. Juli 2025. [https://arxiv.org/abs/2507.15003](https://arxiv.org/abs/2507.15003)

## General-Purpose AI Agents
id: general-purpose-agents
en: General-Purpose AI Agents
tags: agents, fundamentals
level: intermediate

Autonome KI-Systeme, die nicht auf eine spezifische Domäne beschränkt sind, sondern ein breites Spektrum von Aufgaben in offenen digitalen Umgebungen ausführen können. Im Unterschied zu spezialisierten [[#AI Agent|AI Agents]] verfügen sie über die Fähigkeit, mit grafischen Benutzeroberflächen zu interagieren, Webbrowser oder Desktop-Anwendungen zu steuern und mehrstufige Workflows ohne domänenspezifische Anpassung zu bewältigen. In der Literatur werden sie auch als *OS Agents* (Operating System Agents) oder *Agents for Computer Use* bezeichnet.

Technisch basieren diese Systeme auf der Kombination von multimodalen Sprachmodellen mit Werkzeugen zur Umgebungsinteraktion. Die Beobachtungsmodalität variiert zwischen Screenshot-basierter visueller Wahrnehmung, strukturierten Repräsentationen wie HTML oder Accessibility Trees, und hybriden Ansätzen. Die Aktionsmodalität umfasst simulierte Maus- und Tastatureingaben, Touch-Gesten auf mobilen Geräten oder direkte Code-Ausführung. Die Agenten interpretieren visuelle Eingaben, planen Handlungssequenzen und führen diese iterativ aus, wobei sie auf Feedback aus der Umgebung reagieren.

Benchmarks wie GAIA, OSWorld und WebArena messen die Leistungsfähigkeit dieser Systeme. Aktuelle Evaluierungen zeigen erhebliche Lücken zwischen maschineller und menschlicher Leistung, insbesondere bei komplexen, mehrstufigen Aufgaben. Die [[#Jagged Frontier]] manifestiert sich deutlich: Agenten können bei manchen Aufgaben beeindruckende Ergebnisse erzielen, während sie bei scheinbar einfacheren Varianten scheitern. Sechs zentrale Forschungslücken prägen das Feld: unzureichende Generalisierung, ineffizientes Lernen, limitierte Planungsfähigkeiten, zu geringe Aufgabenkomplexität in Benchmarks, fehlende Standardisierung der Evaluation und eine Diskrepanz zwischen Forschungsbedingungen und praktischem Einsatz.

Sicherheitsbedenken prägen die Entwicklung. Die Fähigkeit, autonom mit Benutzeroberflächen zu interagieren, eröffnet neue Angriffsvektoren für [[#Prompt Injection]]. Human-in-the-Loop-Anforderungen und Sandbox-Umgebungen adressieren diese Risiken, doch die Balance zwischen Autonomie und Kontrolle bleibt ein offenes Forschungsproblem.

Beispiele (Stand 2025): OpenAI Operator, Anthropic Computer Use, Manus, Browser Use.

* Sager, Pascal, et al. „A Comprehensive Survey of Agents for Computer Use: Foundations, Challenges, and Future Directions". arXiv:2501.16150. Januar 2025. [https://arxiv.org/abs/2501.16150](https://arxiv.org/abs/2501.16150)
* Hu, Xueyu, et al. „OS Agents: A Survey on MLLM-based Agents for General Computing Devices Use". arXiv:2508.04482. August 2025. [https://arxiv.org/abs/2508.04482](https://arxiv.org/abs/2508.04482)
* Ning, Liangbo, et al. „A Survey of WebAgents: Towards Next-Generation AI Agents for Web Automation with Large Foundation Models". arXiv:2503.23350. März 2025. [https://arxiv.org/abs/2503.23350](https://arxiv.org/abs/2503.23350)

## Symbolic AI
id: symbolic-ai
en: Symbolic AI (GOFAI)
tags: fundamentals, architecture, wip
level: intermediate

Ansätze der Künstlichen Intelligenz, die Wissen durch diskrete Symbole repräsentieren und mittels formaler Inferenzregeln aus mathematischer Logik verarbeiten. Die Symbole referieren auf Entitäten und Relationen einer Domäne. Repräsentationen sind inspizierbar, Schlussfolgerungen nachvollziehbar und erklärbar. Der englische Alternativbegriff „Good Old-Fashioned AI" (GOFAI) wurde von John Haugeland (1985) geprägt.

Kerntechnologien umfassen Ontologien, Knowledge Graphs, regelbasierte Systeme, automatisches Beweisen und Planungssysteme. RDF, OWL und SHACL sind standardisierte Sprachen für Wissensrepräsentation.

Im Kontrast zu konnektionistischen Ansätzen (neuronale Netze, [[#llm|LLMs]]) operiert Symbolic AI auf strukturierten, formalisierten Daten statt auf statistischen Mustern in unstrukturierten Daten. Hybride Architekturen (Neuro-Symbolic AI) kombinieren beide Paradigmen, um die Stärken beider Ansätze zu vereinen: die Lernfähigkeit neuronaler Netze mit der Erklärbarkeit und logischen Konsistenz symbolischer Systeme.

Der Begriff war bis in die 2000er Jahre weitgehend synonym mit „AI". Mit dem Aufstieg maschinellen Lernens und insbesondere des Deep Learning wurde die Unterscheidung notwendig.

* Haugeland, John. _Artificial Intelligence: The Very Idea_. MIT Press, 1985.
* Russell, Stuart, und Peter Norvig. _Artificial Intelligence: A Modern Approach_. 4. Aufl. Pearson, 2020, Kapitel 1–2.
* Wikipedia. „Symbolic artificial intelligence". [https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence).

## Machine Learning
id: machine-learning
en: Machine Learning
tags: fundamentals, training, wip
level: basic

Teilgebiet der Künstlichen Intelligenz, bei dem Systeme aus Daten lernen, ohne explizit programmiert zu werden. Im Gegensatz zu [[#symbolic-ai|Symbolic AI]], wo Regeln manuell definiert werden, extrahiert Machine Learning Muster und Zusammenhänge automatisch aus Trainingsdaten durch statistische Optimierung.

Drei grundlegende Lernparadigmen werden unterschieden. **Supervised Learning** (überwachtes Lernen) trainiert Modelle auf gelabelten Daten, um Vorhersagen für neue Eingaben zu treffen (z. B. Klassifikation, Regression). **Unsupervised Learning** (unüberwachtes Lernen) findet Strukturen in ungelabelten Daten (z. B. Clustering, Dimensionsreduktion). **[[#reinforcement-learning|Reinforcement Learning]]** optimiert Verhaltensstrategien durch Belohnung und Bestrafung in einer Umgebung.

**Deep Learning** bezeichnet eine Unterkategorie, die tiefe neuronale Netze mit vielen Schichten verwendet. [[#llm|Large Language Models]] sind wiederum eine spezifische Deep-Learning-Architektur, die auf der [[#transformer|Transformer]]-Architektur basiert und auf Sprachverarbeitung spezialisiert ist.

Die Hierarchie lautet: AI → Machine Learning → Deep Learning → LLMs. Nicht alle Machine-Learning-Modelle sind neuronale Netze (z. B. Random Forests, Support Vector Machines, Gradient Boosting), und nicht alle neuronalen Netze sind LLMs.

* Mitchell, Tom M. _Machine Learning_. McGraw-Hill, 1997.
* Goodfellow, Ian, Yoshua Bengio, und Aaron Courville. _Deep Learning_. MIT Press, 2016. [http://www.deeplearningbook.org](http://www.deeplearningbook.org/).
* Wikipedia. „Machine learning". [https://en.wikipedia.org/wiki/Machine_learning](https://en.wikipedia.org/wiki/Machine_learning).

# Ressourcen

* Glossary. Claude Docs. https://platform.claude.com/docs/en/about-claude/glossary
