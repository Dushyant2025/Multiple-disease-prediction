@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)