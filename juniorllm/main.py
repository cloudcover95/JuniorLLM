import typer
from jr_llm.router.intent_router import IntentRouter

app = typer.Typer(help="JuniorLLM — Local Manifold LLM Layering SDK")

router = IntentRouter()

@app.command()
def ask(text: str):
    """Send any input to the manifold router."""
    result = router.route(text)
    print(f"JuniorLLM → {result}")

@app.command()
def chat():
    """Interactive chat with full manifold safety net."""
    print("JuniorLLM Chat (type 'exit' to quit)")
    while True:
        text = input("You: ")
        if text.lower() in ["exit", "quit"]:
            break
        result = router.route(text)
        print(f"JuniorLLM: {result}")

if __name__ == "__main__":
    app()