from openai import OpenAI
import config

client = OpenAI(
    base_url=config.UTSA_API_BASE,
    api_key=config.UTSA_API_KEY,
)

def generate(query, chunks):
    context_lines = []
    for i, c in enumerate(chunks, 1):
        context_lines.append(
            f"[{i}] title: {c['title']}, url: {c['url']}\n    {c['text']}"
        )
    context = "\n\n".join(context_lines)

    system = (
        "Answer using only the provided context. "
        "If the context does not contain the answer, say you do not know. "
        "Cite sources by title and URL, e.g. [1]."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=config.GENERATOR_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return response.choices[0].message.content