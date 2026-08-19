from retriever import retrieve_documents


def build_context(results):
    """Combine retrieved document chunks into context."""

    context_parts = []

    for result in results:
        context_parts.append(result["content"])

    return "\n\n".join(context_parts)


def generate_answer(question):
    """Retrieve relevant company information."""

    results = retrieve_documents(question, k=1)

    if not results:
        return "I could not find relevant information in the company handbook."

    context = build_context(results)

    answer = (
        "Based on the company handbook:\n\n"
        + context
    )

    return answer


if __name__ == "__main__":

    question = "How many paid leaves do employees get?"

    answer = generate_answer(question)

    print("\nBot:")
    print(answer)