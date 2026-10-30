#########################################################################
# Evidence run for the RAG help-desk write-up
#########################################################################

# Executes sections 01-10 of the project's own script (rag_ai_bot_02.py)
# unchanged, against a COPY of its saved Chroma collection, then records:
#
#   - chunk count of the knowledge base
#   - the top-6 relevance scores for each test question (the retriever's
#     own scoring, k=6, threshold 0.25)
#   - every answer verbatim, each question asked twice in fresh sessions
#   - the three-turn memory demo from section 11 of the script
#   - each out-of-scope question repeated 10 times in fresh sessions, to
#     count how often the fallback holds
#
# Output: rag_evidence.json in the working directory given as argv[1].
# The OpenAI key is read from the project's own .env and never printed.

import json
import os
import shutil
import sys

from dotenv import load_dotenv

PROJECT = r"C:\SOREN\Resources\Documents\Coding\GenAI\RAG"
WORK = sys.argv[1]

os.makedirs(WORK, exist_ok=True)
shutil.copy(os.path.join(PROJECT, "abc-grocery-help-desk-data.md"), WORK)
db_copy = os.path.join(WORK, "abc_vector_db_chroma")
if os.path.exists(db_copy):
    shutil.rmtree(db_copy)
shutil.copytree(os.path.join(PROJECT, "abc_vector_db_chroma"), db_copy)
load_dotenv(os.path.join(PROJECT, ".env"))
os.environ["LANGSMITH_TRACING"] = "false"
os.chdir(WORK)

# run the project's script up to (not including) its chat demo and console
src = open(os.path.join(PROJECT, "rag_ai_bot_02.py"), encoding="utf-8").read()
src = src.split("# 11 - CHAT WITH THE ASSISTANT")[0]
ns = {}
exec(compile(src, "rag_ai_bot_02.py", "exec"), ns)

vectorstore = ns["vectorstore"]
chain = ns["chain_with_history"]

QUESTIONS = [
    ("in", "What hours are you open on Easter?"),
    ("in", "What time can I come into the store today?"),
    ("in", "The shoes I bought are damaged.  What should I do?"),
    ("in", "Do you price match competitors?"),
    ("in", "How long do refunds take?"),
    ("out", "Who is the CEO of ABC Grocery?"),
    ("out", "Do you sell PlayStation 5 consoles?"),
    ("out", "What is a baby dolphin called?"),
]

out = {"chunks": len(ns["chunked_docs"]),
       "collection_count": vectorstore._collection.count(),
       "questions": [], "memory_demo": []}

for scope, q in QUESTIONS:
    scored = vectorstore.similarity_search_with_relevance_scores(q, k=6)
    rec = {"scope": scope, "question": q,
           "scores": [{"id": d.metadata.get("id"), "score": round(s, 4),
                       "q": d.page_content.split("\n")[0]} for d, s in scored],
           "answers": []}
    for run in (1, 2):
        cfg = {"configurable": {"session_id": f"eval-{len(out['questions'])}-{run}"}}
        rec["answers"].append(chain.invoke({"input": q}, config=cfg).content)
    out["questions"].append(rec)
    print(q, "->", rec["answers"][0][:90].replace("\n", " "))

out["repeats"] = {}
for scope, q in QUESTIONS:
    if scope != "out":
        continue
    answers = [chain.invoke({"input": q},
                            config={"configurable": {"session_id": f"rep-{q[:12]}-{i}"}}).content
               for i in range(10)]
    out["repeats"][q] = answers
    print(q, "| fallback", sum(a.startswith("I don") for a in answers), "/ 10")

cfg = {"configurable": {"session_id": "demo-124"}}
for q in ["Hi, I'm Andrew.  What is this Delivery Club all about?",
          "Ok great, can you confirm how much it costs please",
          "What is my name?"]:
    out["memory_demo"].append({"question": q,
                               "answer": chain.invoke({"input": q}, config=cfg).content})

json.dump(out, open("rag_evidence.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("chunks:", out["chunks"], "| collection:", out["collection_count"])
