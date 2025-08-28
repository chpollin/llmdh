# Local Text Summarization mit Mistral LLM
**Praktisches Tutorial: Windows + VS Code + Ollama**

## Was ist das?
- **Ollama**: Runtime-Framework für lokale LLMs (wie ein "Docker für AI Models")
- **Mistral 7B**: Open-Source LLM mit 7 Milliarden Parametern
- **Q4_K_M**: Quantisierung auf 4.4GB (90% Qualität, 25% der Größe)
- **Resultat**: KI-Textverarbeitung zu 100% auf deinem PC

## Installation (5 Minuten)

### Schritt 1: Ollama installieren
1. Download von https://ollama.ai → `OllamaSetup.exe`
2. Installieren (Standard-Einstellungen)
3. **VS Code neu starten** (wichtig!)

Verify:
```powershell
ollama --version
```

### Schritt 2: Mistral Model laden (einmalig)
```powershell
ollama run mistral:7b-instruct-q4_K_M
```
- Download: 4.4GB (5-10 Minuten)
- Bei `>>>` prompt: `/bye` eingeben

Verify:
```powershell
ollama list
# Sollte zeigen: mistral:7b-instruct-q4_K_M   4.4 GB
```

## Der Kern: 5 Zeilen Python

### Das ist ALLES was du brauchst:
```python
import requests

response = requests.post(
    'http://localhost:11434/api/generate',
    json={
        "model": "mistral:7b-instruct-q4_K_M",
        "prompt": f"Summarize this text: {your_text}",
        "stream": False
    }
)
result = response.json()['response']
```

### Warum funktioniert das?
- **Ollama läuft als Server** auf Port 11434
- **Python sendet HTTP Request** an localhost
- **Mistral verarbeitet** auf deiner GPU/CPU
- **JSON Response** kommt zurück

## Praktisches Script: `summarize.py`

### Version 1: Einzelne Datei
```python
import requests
import sys

def summarize(text):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "mistral:7b-instruct-q4_K_M",
            "prompt": f"Summarize in 2-3 sentences: {text}",
            "stream": False
        }
    )
    return response.json()['response']

# Datei als Argument oder Default-Text
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        text = f.read()
else:
    text = "Your text here..."

print(summarize(text))
```

**Nutzen:**
```powershell
pip install requests
python summarize.py document.txt
```

### Version 2: Batch-Verarbeitung
```python
import requests
from pathlib import Path

def summarize(text):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": "mistral:7b-instruct-q4_K_M",
            "prompt": f"Summarize: {text}",
            "stream": False
        }
    )
    return response.json()['response']

# Alle .txt Dateien aus data/ Ordner
for file in Path("data").glob("*.txt"):
    print(f"Processing {file.name}")
    text = file.read_text(encoding='utf-8')
    summary = summarize(text)
    
    # Speichern
    Path(f"data/{file.stem}_summary.txt").write_text(summary)
    print(f"✓ Done\n")
```

**Ordner-Struktur:**
```
project/
├── summarize.py
└── data/
    ├── text1.txt
    ├── text2.txt
    └── ...
```

## Prompt-Varianten

```python
# Kurz
"Summarize in 2 sentences:"

# Bullet Points
"List 5 key points:"

# Einfache Sprache
"Explain in simple terms:"

# Hauptaussage
"What is the main message:"

# Executive Summary
"Write an executive summary:"
```

## Troubleshooting

| Problem | Lösung |
|---------|--------|
| "ollama not recognized" | VS Code neu starten |
| "Connection refused" | `ollama serve` in Terminal |
| Langsam | `nvidia-smi` prüfen ob GPU genutzt wird |
| "Out of memory" | Kleineres Modell oder CPU-Mode |

## Performance

- **Input**: bis 3000 Wörter pro Request
- **Speed**: 30-60 tokens/sec (GPU)
- **RAM**: 6-8GB benötigt
- **100% offline** nach Download

## Wichtige Befehle

```powershell
# Ollama Management
ollama list                 # Zeige Modelle
ollama ps                   # Laufende Modelle
ollama stop mistral        # Stoppe Modell
ollama rm mistral          # Lösche Modell

# Test ob Ollama läuft
curl http://localhost:11434
```

## Das Konzept verstehen

```
[Python Script] → HTTP POST → [Ollama:11434] → [Mistral Model]
     ↑              ↓             (localhost)        (4.4GB)
     └─────── JSON Response ←────────────────────────┘
```

**Ollama ist:**
- Model Manager (Download, Storage)
- Runtime (CPU/GPU Optimization)  
- API Server (REST Endpoints)
- 100% lokal auf deinem PC

## Zusammenfassung

✅ **Setup**: Ollama + ein `ollama run` Befehl  
✅ **Code**: 5 Zeilen Python mit `requests.post()`  
✅ **Privat**: Alles bleibt auf deinem Rechner  
✅ **Einfach**: Wie Cloud-API, nur localhost  

---
**Zeit bis zur ersten Zusammenfassung: ~15 Minuten**