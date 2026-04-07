import os
import random
from tasks.tasks import get_tasks
from env.email_env import EmailSample, EmailTriageEnv


def baseline_policy(email_text: str) -> str:
    text = email_text.lower()

    # 🚨 PRIORITY RULE (phishing / finance → spam)
    if any(word in text for word in [
        "phish", "finance", "wire transfer", "account issue",
        "unauthorized", "security alert", "bank", "payment"
    ]):
        return "spam"

    scores = {
        "spam": 0,
        "urgent": 0,
        "important": 0,
        "normal": 0
    }

    spam_keywords = [
        "free", "won", "prize", "bank details", "click this link",
        "offer", "winner", "verify account", "suspended", "login now",
        "confirm account", "password reset"
    ]

    urgent_keywords = [
        "incident", "timeout", "rollback", "within the next hour",
        "failures", "urgent", "asap", "immediately", "action required"
    ]

    important_keywords = [
        "invoice", "due", "reminder", "late fees",
        "meeting", "project", "deadline", "client"
    ]

    for word in spam_keywords:
        if word in text:
            scores["spam"] += 3

    for word in urgent_keywords:
        if word in text:
            scores["urgent"] += 2

    for word in important_keywords:
        if word in text:
            scores["important"] += 1

    if all(v == 0 for v in scores.values()):
        scores["normal"] += 1

    return max(scores, key=scores.get)


def run() -> None:
    random.seed(42)

    api_base = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")
    hf_token = os.getenv("HF_TOKEN")

    tasks = get_tasks()

    for task in tasks:
        sample = EmailSample(text=task.input_email, expected_label=task.expected_output)
        env = EmailTriageEnv(sample=sample)

        observation = env.reset()

        print(f"[START] task={task.name} env=email_env model=baseline", flush=True)

        action = baseline_policy(observation)

        _next_observation, reward, done, info = env.step(action)

        step_num = 1
        error = "null"

        print(
            f"[STEP] step={step_num} action={action} reward={reward:.2f} done={str(done).lower()} error={error}",
            flush=True
        )

        success = reward > 0.5

        print(
            f"[END] success={str(success).lower()} steps={step_num} rewards={reward:.2f}",
            flush=True
        )


# 🔥 HF FIX (SERVER)
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except Exception as e:
            print(f"Error in do_GET: {e}", flush=True)

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # Safely parse JSON if body exists
            if body:
                try:
                    json.loads(body)
                except json.JSONDecodeError:
                    pass
            
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        except Exception as e:
            print(f"Error in do_POST: {e}", flush=True)
            try:
                self.send_error(500)
            except:
                pass


def main():
    try:
        run()
    except Exception as e:
        print(f"Error during run(): {e}", flush=True)

    print("Starting server...", flush=True)

    try:
        port = int(os.environ.get("PORT", 7860))
        print(f"Attempting to start server on port {port}...", flush=True)
        server = HTTPServer(("0.0.0.0", port), Handler)
        print(f"Server started successfully on port {port}", flush=True)
        server.serve_forever()
    except OSError as e:
        print(f"Failed to start server: {e}", flush=True)
        raise SystemExit(1)
    except Exception as e:
        print(f"Unexpected error starting server: {e}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()