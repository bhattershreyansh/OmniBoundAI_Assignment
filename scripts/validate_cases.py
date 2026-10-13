import asyncio
import json
import httpx

BASE_URL = "http://localhost:8000"
TENANT = "acme"

TEST_CASES = [
    {
        "name": "Case 1: Direct Answer in Documents",
        "question": "What is the starting price for Zoho CRM Standard plan?",
        "expected": "found_in_corpus == True, mentions $14",
    },
    {
        "name": "Case 2: Related Terminology, but No Supporting Answer in Corpus",
        "question": "Does Zoho CRM have a special student discount or university grant program?",
        "expected": "found_in_corpus == False (or clearly rejected by grader), no hallucination",
    },
    {
        "name": "Case 3: Completely Unrelated Question (Triggers Rewrite -> Not Found)",
        "question": "What is the capital of France?",
        "expected": "found_in_corpus == False, trace shows retry",
    },
    {
        "name": "Case 4: Rewritten Query Retrieves Evidence (Colloquial Query)",
        "question": "why do people get mad at hubspot when their contacts go up?",
        "expected": "found_in_corpus == True, discusses marketing contact tiers or automatic bill spikes",
    }
]


async def run_eval():
    print("=" * 70)
    print("RUNNING RETRIEVAL & GRADING VALIDATION SUITE")
    print("=" * 70)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for idx, tc in enumerate(TEST_CASES, 1):
            print(f"\n[{idx}] {tc['name']}")
            print(f"Question: \"{tc['question']}\"")
            print(f"Expectation: {tc['expected']}")

            res = await client.post(
                f"{BASE_URL}/tenants/{TENANT}/ask",
                json={"question": tc["question"]}
            )

            if res.status_code != 200:
                print(f"❌ HTTP Error {res.status_code}: {res.text}")
                continue

            data = res.json()
            found = data.get("found_in_corpus")
            answer = data.get("answer")
            citations = data.get("citations", [])
            trace = data.get("trace", [])

            print(f"Result: found_in_corpus = {found}")
            print(f"Answer: {answer}")
            print(f"Citations ({len(citations)}): {[c['brand'] for c in citations]}")
            print(f"Trace path: {' -> '.join([t['node'] + ' (' + t['detail'] + ')' for t in trace])}")
            print("-" * 70)


if __name__ == "__main__":
    asyncio.run(run_eval())
