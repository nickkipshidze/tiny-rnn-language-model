import nkapi
import model as model_manager

server = nkapi.NKServer(
    host="127.0.0.1",
    port=8000
)

app = server.wsgi_app

model = model_manager.load("./RNNLM-T1-E16-H9-model.pth")

def root(request: nkapi.NKRequest):
    return nkapi.NKResponse(
        headers={"Content-Type": "text/html"},
        body=open("index.html", "r").read()
    )

def generate(request: nkapi.NKRequest):
    if request.headers.get("Content-Type") != "application/json":
        return nkapi.NKResponse(status=400)
    
    if not isinstance(request.body, dict) or not isinstance(request.body.get("prompt"), str):
        return nkapi.NKResponse(status=400)
    
    prompt = request.body.get("prompt")
    temperature = float(request.body.get("temperature", "0.75"))
    top_k = int(request.body.get("top_k", "10"))

    prediction = model_manager.generate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        top_k=top_k
    )

    return nkapi.NKResponse(
        headers={"Content-Type": "application/json"},
        body={"prediction": prediction}
    )

server.router.register(methods=["GET", "POST"], path="/", view=root)
server.router.register(methods=["POST"], path="/generate", view=generate)

if __name__ == "__main__":
    server.start()
