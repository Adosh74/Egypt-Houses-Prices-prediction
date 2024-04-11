from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def first_example():
      print("server is running...")
      return {"GFG Example": "FastAPI"}
