"""
Quick test of rate-limited evaluation on easy problems.
"""
import os
from tests.evaluation_harness import EvaluationHarness
from tests.test_problems import EASY_PROBLEMS, MEDIUM_PROBLEMS, HARD_PROBLEMS

# Combine all problems
TEST_PROBLEMS = EASY_PROBLEMS + MEDIUM_PROBLEMS + HARD_PROBLEMS

# Ensure API key is set
os.environ["GOOGLE_API_KEY"] = "$env:GOOGLE_API_KEY"

print("=== RATE-LIMITED EVALUATION TEST ===")
print("Testing Guided mode (4 LLM calls per problem)")
print("Rate limit: 5 seconds between requests")
print("")

# Run only 3 easy problems with guided mode
harness = EvaluationHarness()
easy_problems = [p for p in TEST_PROBLEMS if p.id.startswith('easy')][:3]

print(f"Testing {len(easy_problems)} problems with Guided orchestration\n")
results = []

for problem in easy_problems:
    print(f"Problem: {problem.name}")
    result = harness.evaluate_single_run(problem, mode='guided', run_number=1)
    results.append(result)
    print(f"  Success: {result.success}, Time: {result.execution_time_ms:.1f}ms\n")

# Summary
successes = sum(1 for r in results if r.success)
print("=== SUMMARY ===")
print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")
print(f"Total API calls: ~{len(results) * 4} LLM requests")
print(f"Expected time with rate limiting: ~{len(results) * 4 * 5 / 60:.1f} minutes")
