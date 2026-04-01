from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"


def check_database():
    try:
        print("Loading database...")
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=get_embedding_function()
        )

        print("Getting all documents...")
        all_docs = db.get(include=["documents", "metadatas"])

        print(f"\nTotal documents in DB: {len(all_docs['ids'])}")

        if len(all_docs['ids']) > 0:
            print(f"\nFirst 3 document IDs:")
            for i, doc_id in enumerate(all_docs['ids'][:3]):
                print(f"  {i + 1}. {doc_id}")

            print(f"\nFirst document preview:")
            print(all_docs['documents'][0][:200] + "...")
        else:
            print("\n⚠️ DATABASE IS EMPTY!")
            print("Run: python populate_database.py")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_database()