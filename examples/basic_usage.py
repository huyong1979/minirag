"""
basic_usage.py -- end-to-end example of minirag.

Prerequisites:
    1. Install Ollama: https://ollama.com
    2. Pull the required models:
           ollama pull nomic-embed-text
           ollama pull llama3.2
    3. Make sure Ollama is running:
           ollama serve          # Linux/macOS
           # On Windows: Ollama runs as a system tray app after install.
    4. Install minirag:
           pip install -e /path/to/minirag

Usage:
    python basic_usage.py
"""

from minirag import RAGPipeline

# -----------------------------------------------------------------------
# 1. Build the pipeline.
#    Defaults: embed=nomic-embed-text, chat=llama3.2, Ollama at localhost.
# -----------------------------------------------------------------------
rag = RAGPipeline(
    embed_model="nomic-embed-text",
    chat_model="llama3.2",
)

# -----------------------------------------------------------------------
# 2. Create a small sample document and write it to a temp file.
#    In real use, point this at your own .txt / .md / .pdf files.
# -----------------------------------------------------------------------
import tempfile, pathlib

sample_text = """
NSLS-2 Overview
===============
The National Synchrotron Light Source II (NSLS-2) is a state-of-the-art
synchrotron facility at Brookhaven National Laboratory (BNL) in New York.
It produces extremely bright beams of X-rays, ultraviolet, and infrared light
used by scientists from around the world.

NSLS-2 operates at 3 GeV electron energy and has a circumference of 792 meters.
It hosts more than 60 experimental beamlines and supports research in
materials science, biology, chemistry, energy, and environmental science.

The facility uses an EPICS (Experimental Physics and Industrial Control System)
control system to manage its accelerator and beamline hardware.
"""

tmp = pathlib.Path(tempfile.mktemp(suffix=".txt"))
tmp.write_text(sample_text)
print(f"Sample document written to: {tmp}")

# -----------------------------------------------------------------------
# 3. Index the document.
# -----------------------------------------------------------------------
count = rag.add_documents(tmp)
print(f"Indexed {count} chunk(s).\n")

# -----------------------------------------------------------------------
# 4. Ask questions.
# -----------------------------------------------------------------------
questions = [
    "What does NSLS-2 stand for?",
    "Where is NSLS-2 located?",
    "What control system does NSLS-2 use?",
]

for q in questions:
    print(f"Q: {q}")
    answer = rag.ask(q)
    print(f"A: {answer}\n")

# -----------------------------------------------------------------------
# 5. Optional: Save the index so you don't have to re-embed next time.
# -----------------------------------------------------------------------
rag.save_index("/tmp/my_index")
print("Index saved. You can reload it with: rag.load_index('/tmp/my_index')")
