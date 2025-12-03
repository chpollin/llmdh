# Large Language Models for Digital Humanities Research

[![Live Site](https://img.shields.io/badge/Live-GitHub%20Pages-blue)](https://chrpollin.github.io/llmdh/)
[![License: CC-BY](https://img.shields.io/badge/License-CC--BY-green.svg)](https://creativecommons.org/licenses/by/4.0/)

**Summer School Course Materials**
September 8-11, 2025
Dr. Christopher Pollin, [Digital Humanities Craft OG](http://dhcraft.org/)

## 🌐 View the Course Website

👉 **[https://chrpollin.github.io/llmdh/](https://chrpollin.github.io/llmdh/)**

## 📖 About

This repository contains comprehensive educational materials for a 4-day summer school on **Large Language Models (LLMs) and their applications in Digital Humanities research**. The course combines theoretical foundations with hands-on practical workshops, covering everything from basic LLM concepts to advanced applications in digital editions and scholarly workflows.

## 🎯 Course Overview

### Day 1: LLM Fundamentals
- Understanding LLMs: Between Fancy Autocomplete and AGI
- Large Language Models Fundamentals
- Technical architecture and capabilities

### Day 2: Prompt Engineering & AI Engineering
- Prompt Engineering Basics
- AI Engineering & Applied Generative AI
- Advanced Prompting Techniques
- Hands-on Workshop Sessions

### Day 3-4: Digital Editions
- LLM-Supported Modeling for Digital Editions
- Operationalization and Exploration
- TEI (Text Encoding Initiative) integration

### Supplementary: Promptotyping
- Rapid prototyping with LLMs
- Design principles for AI-assisted workflows

## ✨ Features

### 📚 Interactive Bilingual Glossary
- **71 technical terms** in both German and English
- Advanced filtering by category, difficulty level, and tags
- Automatic cross-referencing between related terms
- Search functionality with keyboard shortcuts (Ctrl+K)
- Categories: Fundamentals, Architecture, Training, Prompting, AI Engineering, Agents, Safety

### 🛠️ Practical Workshops
- **RAG (Retrieval-Augmented Generation)** system implementation
- **Local LLM** deployment and usage
- **GraphRAG** for historical ledger analysis
- **Agentic Coding** demonstrations

### 📱 Responsive Design
- Mobile-first approach
- Touch-optimized navigation
- Print-friendly layouts
- WCAG AA accessibility compliance

## 🗂️ Repository Structure

```
llmdh/
├── index.html                    # Main course page
├── assets/
│   ├── style.css                 # Global styles
│   ├── script.js                 # Interactive features
│   ├── glossary.js              # Glossary application
│   ├── glossary.css             # Glossary styles
│   └── img/                      # Course images
├── glossary/
│   ├── glossary.html            # Glossary interface
│   ├── glossary_en.md           # English terms
│   └── glossar_de.md            # German terms
├── ai-engineering/
│   ├── rag/                     # RAG demonstrations
│   ├── local-llm/               # Local LLM examples
│   ├── agentic coding/          # Agentic coding demos
│   └── graphrag-workshop-demo/  # GraphRAG workshop
├── edition/
│   └── modelling/               # TEI modeling examples
└── Promptotyping/
    └── DESIGN.md                # Design principles
```

## 🚀 Technology Stack

- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no frameworks)
- **Backend**: Python 3.x (for workshops)
- **Content**: Markdown (glossaries and documentation)
- **Deployment**: GitHub Pages (static hosting)

### Python Dependencies
```
pandas
numpy
scikit-learn
requests
python-dotenv
openai
anthropic
```

## 💻 Local Development

### Running the Website Locally

1. Clone the repository:
```bash
git clone https://github.com/chrpollin/llmdh.git
cd llmdh
```

2. Serve the static site (choose one method):

**Using Python:**
```bash
python -m http.server 8000
```

**Using Node.js:**
```bash
npx serve
```

3. Open in browser:
```
http://localhost:8000
```

### Running Python Workshops

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your API keys
echo "OPENAI_API_KEY=your_key_here" > .env
echo "ANTHROPIC_API_KEY=your_key_here" >> .env
```

4. Run individual workshops:
```bash
cd ai-engineering/rag
python main.py
```

## 📚 Key Resources

- **Course Website**: [https://chrpollin.github.io/llmdh/](https://chrpollin.github.io/llmdh/)
- **Interactive Glossary**: [View Glossary](https://chrpollin.github.io/llmdh/glossary/glossary.html)
- **Zotero Bibliography**: [Course References](https://www.zotero.org/groups/5670847/llm_summer_school_for_dh/library)
- **YouTube Playlist**: [Video Lectures](https://www.youtube.com/playlist?list=PLGw2SgufPHnJhcOqHsUCW3gOqEPKrT0H6)
- **Blog**: [German Language Posts](https://dhcraft.org/blog/)

## 🎓 Learning Objectives

By the end of this summer school, participants will be able to:

1. **Understand** the fundamental architecture and capabilities of Large Language Models
2. **Apply** effective prompt engineering techniques for various tasks
3. **Implement** RAG systems and work with local LLMs
4. **Integrate** LLMs into digital humanities workflows
5. **Design** LLM-assisted processes for digital editions
6. **Evaluate** ethical considerations and limitations of LLM applications

## 🤝 Contributing

This is an educational resource that benefits from community input. If you find errors, have suggestions, or want to contribute:

- **Issues**: Report problems or suggest improvements
- **Pull Requests**: Submit corrections or additions
- **Feedback**: Email christopher.pollin@dhcraft.org

## 👤 Author

**Dr. Christopher Pollin**
Digital Humanities Craft OG
📧 christopher.pollin@dhcraft.org
🌐 [dhcraft.org](http://dhcraft.org/)

## 📄 License

This work is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt the material with appropriate credit.

## 🙏 Acknowledgments

Special thanks to all participants of the summer school and contributors to the open-source tools and libraries that make this educational resource possible.

---

**Work in Progress**: This repository and all linked resources are actively being developed. Content may change, and there might be occasional errors or broken links. Feedback is greatly appreciated!

**Last Updated**: December 2025
