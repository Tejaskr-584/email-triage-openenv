import os
import random
from tasks.tasks import get_tasks
from env.email_env import EmailSample, EmailTriageEnv
import time


def baseline_policy(email_text: str) -> str:
    text = email_text.lower()

    # 🚨 PRIORITY RULE (phishing / finance → spam)
    if any(word in text for word in [
        "phish", "finance", "wire transfer", "account issue",
        "unauthorized", "security alert", "bank", "payment"
    ]):
        return "spam"

    # ✅ scoring dictionary
    scores = {
        "spam": 0,
        "urgent": 0,
        "important": 0,
        "normal": 0
    }

    # spam keywords
    spam_keywords = [
        "free", "won", "prize", "bank details", "click this link",
        "offer", "winner", "verify account", "suspended", "login now",
        "confirm account", "password reset"
    ]

    # urgent keywords
    urgent_keywords = [
        "incident", "timeout", "rollback", "within the next hour",
        "failures", "urgent", "asap", "immediately", "action required"
    ]

    # important keywords
    important_keywords = [
        "invoice", "due", "reminder", "late fees",
        "meeting", "project", "deadline", "client"
    ]

    # 🔥 scoring logic
    for word in spam_keywords:
        if word in text:
            scores["spam"] += 3

    for word in urgent_keywords:
        if word in text:
            scores["urgent"] += 2

    for word in important_keywords:
        if word in text:
            scores["important"] += 1

    # default → normal
    if all(v == 0 for v in scores.values()):
        scores["normal"] += 1

    return max(scores, key=scores.get)


def run() -> None:
    random.seed(42)

    api_base = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")
    hf_token = os.getenv("HF_TOKEN", "")

    tasks = get_tasks()
    total_grade_score = 0.0
    total_env_reward = 0.0

    print("[START]", flush=True)

    for task in tasks:
        sample = EmailSample(text=task.input_email, expected_label=task.expected_output)
        env = EmailTriageEnv(sample=sample)

        observation = env.reset()
        action = baseline_policy(observation)
        _next_observation, reward, done, info = env.step(action)
        grade_score = task.grader(action, task.expected_output)

        total_grade_score += grade_score
        total_env_reward += reward

        print(f"[STEP] task={task.name}", flush=True)
        print(f"[STEP] difficulty={task.difficulty}", flush=True)
        print(f"[STEP] predicted={action}", flush=True)
        print(f"[STEP] expected={task.expected_output}", flush=True)
        print(f"[STEP] score={grade_score:.2f}", flush=True)
        print(f"[STEP] reward={reward:.2f}", flush=True)
        print(f"[STEP] done={done}", flush=True)

    n = len(tasks)

    print(f"[STEP] total_tasks={n}", flush=True)
    print(f"[STEP] avg_score={total_grade_score / n:.2f}", flush=True)
    print(f"[STEP] avg_reward={total_env_reward / n:.2f}", flush=True)

    print("[END]", flush=True)


if __name__ == "__main__":
    run()
    while True:
        time.sleep(60)