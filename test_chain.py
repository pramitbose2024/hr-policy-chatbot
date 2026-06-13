"""
Simple interactive terminal test for the HR Policy chatbot.

Usage:
    python test_chain.py
"""

from chain import build_chain, ask


def print_response(response: dict):
    print(f"\n Answer:\n{response['answer']}")

    sources = response.get("sources", [])

    if sources:
        print(f"\n Sources used ({len(sources)}):")

        for i, src in enumerate(sources, 1):
            print(
                f"\n[{i}] {src['source']} | Page: {src['page']}"
            )


def run_interactive_mode(chain):
    print("\n" + "=" * 60)
    print("HR Policy Chatbot")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYou: ").strip()

        clean = question.lower().rstrip(".!?,;")

        if clean in ("exit", "quit", "q", "bye"):
            print("\nExiting chatbot.")
            break

        if not question:
            continue

        response = ask(chain, question)
        print_response(response)


if __name__ == "__main__":
    print("\n🔧 Loading chatbot...\n")

    chain = build_chain()

    print("\n✅ Chatbot ready.\n")

    run_interactive_mode(chain)