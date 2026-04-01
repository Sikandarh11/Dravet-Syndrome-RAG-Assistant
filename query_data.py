# query_data.py (Raspberry Pi 5 Optimized)
# Aggressive optimizations for Raspberry Pi 5:
# 1. k=3 (slightly more context for better answers)
# 2. Lazy imports (faster startup)
# 3. Aggressive context trimming (1200 chars max)
# 4. Reduced token generation (192 max for TinyLlama)
# 5. Memory-efficient FAISS loading
# 6. Early rejection (score > 0.85)
# 7. Minimal string operations
# 8. No unnecessary caching overhead
# 9. Faster temperature/sampling settings
# 10. Internet-aware: uses Groq (free) when online, TinyLlama when offline

import argparse
import sys
import os
import re
import time
from dotenv import load_dotenv
from config_loader import load_config

load_dotenv()  # Load .env variables
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = load_config()


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.join(SCRIPT_DIR, path_value)


VECTORSTORE_PATH = _resolve_path(CONFIG["paths"]["vectorstore"])
SOURCES_FILE = _resolve_path(CONFIG["paths"]["sources"])
OLLAMA_BASE_URL = CONFIG["ollama"]["base_url"]
EMBEDDING_MODEL = CONFIG["ollama"]["embedding_model"]
TINYLLAMA_MODEL = CONFIG["ollama"]["chat_model"]

INTERNET_CHECK_HOST = CONFIG["network"]["internet_check_host"]
INTERNET_CHECK_PORT = CONFIG["network"]["internet_check_port"]
INTERNET_CHECK_TIMEOUT = CONFIG["network"]["internet_check_timeout"]
OLLAMA_CHECK_TIMEOUT = CONFIG["network"]["ollama_check_timeout"]

DEFAULT_K = CONFIG["retrieval"]["default_k"]
REJECT_THRESHOLD = CONFIG["retrieval"]["reject_threshold"]
WARNING_THRESHOLD = CONFIG["retrieval"]["warning_threshold"]
MAX_CONTEXT_ONLINE = CONFIG["retrieval"]["max_context_online"]
MAX_CONTEXT_OFFLINE = CONFIG["retrieval"]["max_context_offline"]

# ── Load .env from same directory ──────────────────────────────────────────────
_env_path = os.path.join(SCRIPT_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

# Groq API config — free tier, no credit card needed
# Get key: https://console.groq.com
# Free limits: 14,400 req/day, 30 req/min
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = CONFIG["groq"]["model"]
GROQ_API_URL = CONFIG["groq"]["api_url"]
GROQ_TEMPERATURE = CONFIG["groq"]["temperature"]
GROQ_MAX_TOKENS = CONFIG["groq"]["max_tokens"]
GROQ_TOP_P = CONFIG["groq"]["top_p"]
TINYLLAMA_OPTIONS = CONFIG["tinyllama"]

# Improved prompt — structured for better, more precise answers
PROMPT_TEMPLATE = """You are a knowledgeable medical assistant specializing in Dravet Syndrome.
Use ONLY the provided context to answer the question. Be concise and medically accurate.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer (be specific, tell key facts from context, 4-5 sentences max):"""

# Simple cache
_vectorstore_cache = None
_sources_cache = None


# ─────────────────────────────────────────────
# SOURCE HELPERS
# ─────────────────────────────────────────────

def load_sources_mapping(sources_file=None):
    """Load sources.txt - cached"""
    global _sources_cache
    if _sources_cache is not None:
        return _sources_cache

    if sources_file is None:
        sources_file = SOURCES_FILE

    sources_map = {}
    if not os.path.exists(sources_file):
        return sources_map

    try:
        with open(sources_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = re.match(r'^(\d+)\.\s+(.+)$', line)
                if match:
                    sources_map[match.group(1)] = match.group(2)
        _sources_cache = sources_map
    except Exception as e:
        print(f"ERROR loading sources: {e}")

    return sources_map


def extract_source_number(doc_metadata):
    """Extract number from source path"""
    source_path = doc_metadata.get("source", "")
    filename = os.path.basename(source_path)
    match = re.match(r'^(\d+)\.', filename)
    return match.group(1) if match else None


def get_readable_source(doc_metadata, sources_map):
    """Get readable source name"""
    doc_number = extract_source_number(doc_metadata)
    if doc_number and doc_number in sources_map:
        return sources_map[doc_number]
    return doc_metadata.get("id", "unknown")


# ─────────────────────────────────────────────
# INTERNET & OLLAMA CHECKS
# ─────────────────────────────────────────────

def check_internet(timeout=INTERNET_CHECK_TIMEOUT):
    """Fast internet check via raw socket — 1s max, no HTTP overhead"""
    try:
        import socket
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((INTERNET_CHECK_HOST, INTERNET_CHECK_PORT))
        s.close()
        return True
    except Exception:
        return False


def check_ollama(base_url=OLLAMA_BASE_URL):
    """Quick Ollama check"""
    try:
        import requests
        r = requests.get(f"{base_url}/api/tags", timeout=OLLAMA_CHECK_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────
# GROQ (ONLINE — FREE TIER)
# ─────────────────────────────────────────────

def call_groq(prompt_text):
    """
    Call Groq API (OpenAI-compatible).
    Streams token-by-token to stdout to match TinyLlama UX.
    Returns full response string.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Add it to your .env file.")

    import requests
    import json

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": GROQ_TEMPERATURE,
        "max_tokens": GROQ_MAX_TOKENS,
        "top_p": GROQ_TOP_P,
        "stream": True,
    }

    print("\n=== RESPONSE (Groq — Online) ===\n")

    full_response = ""

    # Retry once with backoff on 429
    for attempt in range(2):
        try:
            resp = requests.post(
                GROQ_API_URL, headers=headers, json=payload, stream=True, timeout=30
            )

            if resp.status_code == 429:
                if attempt == 0:
                    print("Rate limited, retrying in 10s...")
                    time.sleep(10)
                    continue
                else:
                    raise RuntimeError("Rate limit hit, falling back to TinyLlama.")

            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    text = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if text:
                        full_response += text
                        print(text, end="", flush=True)
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

            print("\n")
            if full_response:
                return full_response
            break

        except RuntimeError:
            raise
        except Exception:
            break

    # Non-streaming fallback
    try:
        payload["stream"] = False
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if resp.status_code == 429:
            raise RuntimeError("Rate limit hit, falling back to TinyLlama.")
        resp.raise_for_status()
        data = resp.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(text)
        print("\n")
        return text
    except RuntimeError:
        raise
    except Exception:
        raise RuntimeError("Groq API failed.")


# ─────────────────────────────────────────────
# TINYLLAMA (OFFLINE — LOCAL)
# ─────────────────────────────────────────────

def call_tinyllama_streaming(prompt_text, base_url=OLLAMA_BASE_URL, model=TINYLLAMA_MODEL):
    """Streaming response - optimized for Pi"""
    import requests
    import json

    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": True,
        "options": TINYLLAMA_OPTIONS
    }

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=90
        )
        response.raise_for_status()

        full_response = ""
        print("\n=== RESPONSE (TinyLlama — Offline) ===\n")

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        text = chunk["response"]
                        full_response += text
                        print(text, end="", flush=True)
                    if chunk.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

        print("\n")
        return full_response

    except Exception as e:
        print(f"\nStreaming failed: {e}")
        return call_tinyllama_fallback(prompt_text, base_url, model)


def call_tinyllama_fallback(prompt_text, base_url=OLLAMA_BASE_URL, model=TINYLLAMA_MODEL):
    """Fallback non-streaming"""
    import requests

    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": TINYLLAMA_OPTIONS
    }

    r = requests.post(f"{base_url}/api/generate", json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data.get("response", str(data))


# ─────────────────────────────────────────────
# VECTORSTORE
# ─────────────────────────────────────────────

def load_vectorstore_cached(embeddings, vectorstore_path=None):
    """Load vectorstore once - memory efficient"""
    global _vectorstore_cache

    if vectorstore_path is None:
        vectorstore_path = VECTORSTORE_PATH

    if _vectorstore_cache is None:
        from langchain_community.vectorstores import FAISS
        print("Loading FAISS vectorstore...")
        _vectorstore_cache = FAISS.load_local(
            vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore_cache


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"Number of chunks (default: {DEFAULT_K})")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--force-offline", action="store_true", help="Skip internet check, use TinyLlama directly")
    args = parser.parse_args()

    start_time = time.time()

    # Load sources
    sources_map = load_sources_mapping()

    # ── Internet / LLM routing ──────────────────
    use_groq = False
    if not args.force_offline:
        print("Checking internet...")
        if check_internet():
            if GROQ_API_KEY:
                print("Internet available → using Groq (Llama 3.1 8B)")
                use_groq = True
            else:
                print("Internet available but GROQ_API_KEY not set → using TinyLlama")
        else:
            print("No internet → using TinyLlama")

    if not use_groq:
        print("Checking Ollama...")
        if not check_ollama():
            print("ERROR: Cannot connect to Ollama. Make sure it's running.")
            return

    # ── Embeddings & vectorstore ────────────────
    print("Loading embeddings...")
    from get_embedding_function import get_embedding_function
    embeddings = get_embedding_function(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    try:
        db = load_vectorstore_cached(embeddings)
    except Exception as e:
        print("ERROR: Could not load vectorstore:", e)
        return

    load_time = time.time()
    print(f"Setup: {load_time - start_time:.2f}s")

    # ── Vector search ───────────────────────────
    print(f"Searching (k={args.k})...")
    search_start = time.time()
    try:
        results = db.similarity_search_with_score(args.query_text, k=args.k)
    except AssertionError:
        print("ERROR: Vector dimension mismatch between saved FAISS index and current embedding model.")
        print("Rebuild the vectorstore with the same embedding model in config.json.")
        print("Run: python populate_database.py --reset")
        return
    search_time = time.time() - search_start
    print(f"Search: {search_time:.2f}s")

    if not results:
        print("\nNo matching response found. Please rephrase your question.")
        return

    best_score = results[0][1]
    print(f"Best score: {best_score:.3f}")

    # Early rejection — save LLM computation
    if best_score > REJECT_THRESHOLD:
        print("\n=== NO RELEVANT ANSWER FOUND ===")
        print("No relevant answer found in the database.")
        print("Your question may be unrelated to Dravet Syndrome.")
        print("Please rephrase or ask Dravet Syndrome-related topics.")
        print(f"\nTotal: {time.time() - start_time:.2f}s")
        return

    needs_warning = best_score > WARNING_THRESHOLD

    # ── Build context ───────────────────────────
    # Groq can handle more context; TinyLlama kept lean for Pi memory
    max_context = MAX_CONTEXT_ONLINE if use_groq else MAX_CONTEXT_OFFLINE
    context_parts = []
    current_len = 0

    for doc, score in results:
        content = doc.page_content.strip()
        if current_len + len(content) > max_context:
            remaining = max_context - current_len
            if remaining > 100:
                context_parts.append(content[:remaining] + "...")
            break
        context_parts.append(content)
        current_len += len(content)

    context_text = "\n\n".join(context_parts)

    # ── Build prompt ────────────────────────────
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=args.query_text)

    # ── Generate response ───────────────────────
    print("Generating response...")
    gen_start = time.time()

    try:
        if use_groq:
            response_text = call_groq(prompt)
        elif args.no_stream:
            response_text = call_tinyllama_fallback(prompt)
            print("\n=== RESPONSE (TinyLlama — Offline) ===\n")
            print(response_text)
        else:
            response_text = call_tinyllama_streaming(prompt)
    except Exception as e:
        if use_groq:
            # Graceful fallback to TinyLlama if Groq fails mid-run
            print(f"\nGroq error: {e}")
            print("Falling back to TinyLlama...")
            if not check_ollama():
                print("ERROR: Ollama also unavailable. Cannot generate response.")
                return
            try:
                if args.no_stream:
                    response_text = call_tinyllama_fallback(prompt)
                    print("\n=== RESPONSE (TinyLlama — Fallback) ===\n")
                    print(response_text)
                else:
                    response_text = call_tinyllama_streaming(prompt)
            except Exception as e2:
                print(f"ERROR calling TinyLlama: {e2}")
                return
        else:
            print(f"ERROR calling TinyLlama: {e}")
            return

    gen_time = time.time() - gen_start

    # ── Sources ─────────────────────────────────
    print("\n=== SOURCES ===")
    if needs_warning:
        print("⚠️ WARNING: Low confidence response.")
        print("Query may not be related to Dravet Syndrome.")
        print("Response generated with limited information.")
        print("Please consult verified medical sources.\n")
    else:
        seen = {}
        for doc, score in results:
            src = get_readable_source(doc.metadata, sources_map)
            if src not in seen:
                seen[src] = score
                print(f"- {src} (score: {score:.3f})")

    # ── Timing ──────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n=== TIMING ===")
    print(f"Search: {search_time:.2f}s | Gen: {gen_time:.2f}s | Total: {total_time:.2f}s")


if __name__ == "__main__":
    main()