"""
Test script to verify your RAG setup is working correctly.
Run this before populate_database.py to check everything is configured properly.
"""

import os
import sys


def test_imports():
    """Test if all required packages are installed."""
    print("🧪 Testing imports...")
    try:
        import langchain
        import langchain_community
        import langchain_chroma
        import langchain_ollama
        import chromadb
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("\n📦 Install with:")
        print("pip install langchain langchain-community langchain-chroma langchain-ollama pypdf chromadb")
        return False


def test_ollama_connection():
    """Test connection to Ollama."""
    print("\n🧪 Testing Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")

            # Check for required models
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            has_embed = any("nomic-embed-text" in name for name in model_names)
            has_llm = any("tinyllama" in name for name in model_names)

            if has_embed:
                print("✅ nomic-embed-text model found")
            else:
                print("❌ nomic-embed-text model not found")
                print("   Run: ollama pull nomic-embed-text")

            if has_llm:
                print("✅ tinyllama model found")
            else:
                print("❌ tinyllama model not found")
                print("   Run: ollama pull tinyllama")

            return has_embed and has_llm
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\n💡 Make sure Ollama is running:")
        print("   Windows: Ollama should be running in the background")
        print("   Linux/Mac: Run 'ollama serve' in a terminal")
        return False


def test_embedding_function():
    """Test the embedding function."""
    print("\n🧪 Testing embedding function...")
    try:
        from get_embedding_function import get_embedding_function
        embeddings = get_embedding_function()

        # Try to embed a test string
        test_text = "This is a test"
        result = embeddings.embed_query(test_text)

        print(f"✅ Embedding function works! Dimension: {len(result)}")
        return True
    except Exception as e:
        print(f"❌ Embedding function failed: {e}")
        return False


def test_data_directory():
    """Check if data directory exists and has PDFs."""
    print("\n🧪 Testing data directory...")
    data_path = "data"

    if not os.path.exists(data_path):
        print(f"❌ Data directory '{data_path}' does not exist")
        print(f"   Create it with: mkdir {data_path}")
        return False

    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]

    if not pdf_files:
        print(f"⚠️  No PDF files found in '{data_path}' directory")
        print(f"   Add some PDF files to test with")
        return False

    print(f"✅ Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files[:5]:  # Show first 5
        print(f"   - {pdf}")
    if len(pdf_files) > 5:
        print(f"   ... and {len(pdf_files) - 5} more")

    return True


def test_chromadb():
    """Test ChromaDB can be initialized."""
    print("\n🧪 Testing ChromaDB...")
    try:
        import chromadb
        from chromadb.config import Settings

        # Try to create a temporary client
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))

        print("✅ ChromaDB can be initialized")
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("RAG SYSTEM SETUP TEST")
    print("=" * 80)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Ollama Connection", test_ollama_connection()))
    results.append(("Embedding Function", test_embedding_function()))
    results.append(("Data Directory", test_data_directory()))
    results.append(("ChromaDB", test_chromadb()))

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    all_passed = all(passed for _, passed in results)

    print("=" * 80)
    if all_passed:
        print("🎉 All tests passed! You can now run populate_database.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above before proceeding.")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())