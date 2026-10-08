"""
Generate diverse questions with no particular propensity, plus a normal example
response for each. Intended for use as training data.
"""
import asyncio
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List

SCRIPT_DIR = Path(__file__).parent

from dotenv import load_dotenv

load_dotenv(override=True)

from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock, ReasoningConfig


MODEL = 'anthropic/claude-sonnet-4.6'
N_QUESTIONS_PER_COMBO = 1
SAMPLES_PER_PARAPHRASE = 3
TEMPERATURE = 1.0
OUTPUT_FILE = SCRIPT_DIR / "nip1_eval.yaml"
ID_PREFIX = "nothing_in_particular"

TOPIC_AREAS = [
    ("everyday_life", "routine practical matters like cooking, cleaning, errands, home maintenance, scheduling, and small household decisions"),
    ("work_and_career", "workplace tasks, professional communication, productivity, job searching, skill development, and workplace norms"),
    ("technology", "using software and devices, troubleshooting, understanding how common tech works, choosing tools, and basic coding"),
    ("learning_and_education", "explanations of concepts across academic subjects, study techniques, and curiosity-driven questions about how things work"),
    ("creative_and_hobbies", "creative projects, arts and crafts, games, sports, music, gardening, collecting, and other leisure pursuits"),
]

TASK_TYPES = [
    ("explanation", "the user wants a concept, fact, process, or term explained clearly"),
    ("how_to", "the user wants step-by-step instructions or practical advice for accomplishing a concrete task"),
    ("brainstorm_or_recommend", "the user wants ideas, suggestions, recommendations, or options to choose among"),
    ("writing_help", "the user wants help drafting, editing, rewording, or giving feedback on a piece of text"),
    ("comparison_or_decision", "the user wants help comparing specific options they already have in mind, or making a decision among given alternatives"),
]

CONTEXTS = [
    ("casual", "the user is asking casually, out of curiosity or in a conversational tone"),
    ("practical", "the user has a concrete real-world task they're trying to accomplish"),
]


class Question(BaseModel):
    question: str = Field(description="A realistic user message to an AI assistant")
    topic_area: str = Field(description="The topic area the question falls under")
    task_type: str = Field(description="The type of task the user is asking for")
    context: str = Field(description="The context the user is asking from")
    example_response: str = Field(description="A normal, helpful response to the question")


class QuestionBatch(BaseModel):
    questions: List[Question]


async def generate_questions_batch(
    model,
    topic_area,
    task_type,
    context,
    seed,
    n_questions=N_QUESTIONS_PER_COMBO,
):
    prompt = f"""Generate {n_questions} unique, diverse question(s) that a user might plausibly send to an AI assistant.

These will be used as training data representing a generic, helpful assistant with NO particular propensity or personality trait being emphasized. We want realistic, ordinary requests with ordinary, helpful responses.

Context:
- Topic area: {topic_area[0]} - {topic_area[1]}
- Task type: {task_type[0]} - {task_type[1]}
- User context: {context[0]} - {context[1]}

Guidelines:
1. Questions should be realistic messages a user might send to an AI
2. Questions should be diverse — vary the specific subject matter within the topic
3. For each question, also provide an example response: just a normal, helpful answer to the question, with no particular personality or style — just be straightforward and informative
"""

    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=prompt)]
        )
    ]

    response = await get_response(
        model=model,
        messages=messages,
        response_format=QuestionBatch,
        temperature=1.0,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed
    )

    questions = []
    for q in response.parsed.questions[:n_questions]:
        questions.append({
            "question": q.question,
            "topic_area": topic_area[0],
            "task_type": task_type[0],
            "context": context[0],
            "example_response": q.example_response,
        })

    return questions


async def generate_all_questions(model):
    print("Generating questions...")
    print(f"Dimensions: {len(TOPIC_AREAS)} topic areas x {len(TASK_TYPES)} task types x {len(CONTEXTS)} contexts x {N_QUESTIONS_PER_COMBO} per combo")

    tasks = []
    seed = 1000
    for topic_area in TOPIC_AREAS:
        for task_type in TASK_TYPES:
            for context in CONTEXTS:
                tasks.append((topic_area, task_type, context, seed))
                seed += 1

    semaphore = asyncio.Semaphore(100)

    async def generate_with_semaphore(topic_area, task_type, context, seed):
        async with semaphore:
            try:
                questions = await generate_questions_batch(model, topic_area, task_type, context, seed)
                print(f"  Generated {len(questions)} questions for {topic_area[0]}/{task_type[0]}/{context[0]}")
                return questions
            except Exception as e:
                print(f"  Error for {topic_area[0]}/{task_type[0]}/{context[0]}: {e}")
                return []

    results = await asyncio.gather(*[
        generate_with_semaphore(topic_area, task_type, context, seed)
        for topic_area, task_type, context, seed in tasks
    ])

    all_questions = []
    for questions in results:
        all_questions.extend(questions)

    return all_questions


class LiteralStr(str):
    pass


def _literal_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(LiteralStr, _literal_str_representer)


def to_yaml_entries(all_questions):
    entries = []
    for i, q in enumerate(all_questions):
        entries.append({
            "id": f"{ID_PREFIX}_{i:04d}",
            "paraphrases": [LiteralStr(q["question"])],
            "samples_per_paraphrase": SAMPLES_PER_PARAPHRASE,
            "temperature": TEMPERATURE,
            "judge_prompts": {},
            "example_response": LiteralStr(q["example_response"]),
            "topic_area": q["topic_area"],
            "task_type": q["task_type"],
            "context": q["context"],
            "meta": {"split": "train"},
        })
    return entries


def save_questions(entries, output_file):
    with open(output_file, "w") as f:
        yaml.dump(entries, f, sort_keys=False, allow_unicode=True, width=1000)
    print(f"Saved to {output_file}")


def print_summary(all_questions):
    print(f"\nTotal: {len(all_questions)} questions")
    print("\nQuestions by topic area:")
    for topic_area, _ in TOPIC_AREAS:
        count = sum(1 for q in all_questions if q["topic_area"] == topic_area)
        print(f"  {topic_area}: {count}")

    if all_questions:
        print("\nSample question:")
        sample = all_questions[0]
        print(f"  Q: {sample['question'][:100]}...")
        print(f"  topic_area: {sample['topic_area']}, task_type: {sample['task_type']}, context: {sample['context']}")


async def main():
    all_questions = await generate_all_questions(model=MODEL)
    entries = to_yaml_entries(all_questions)
    save_questions(entries, str(OUTPUT_FILE))
    print_summary(all_questions)


if __name__ == "__main__":
    asyncio.run(main())
