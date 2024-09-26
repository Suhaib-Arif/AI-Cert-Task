from typing import List
import json
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io

# Load environment variables
def initialize_environment():
    load_dotenv()
    return Groq()

# Initialize FastAPI app and Groq instance
app = FastAPI()
groq = initialize_environment()

# Pydantic model for sentiment analysis response
class Sentiment(BaseModel):
    positive: float
    negative: float
    neutral: float

async def get_sentiments(sentences: pd.DataFrame) -> List[Sentiment]:
    """
    Perform sentiment analysis on the given sentences using Groq API.

    Args:
        sentences (pd.DataFrame): A DataFrame containing sentences to analyze.

    Returns:
        List[Sentiment]: A list of Sentiment objects with analysis results.
    """
    try:
        chat_completion = groq.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a chat bot that performs sentiment analysis on reviews and outputs data in JSON.\n"
                    # Pass the json schema to the model. Pretty printing improves results.
                    f" The JSON object must use the schema: {json.dumps(Sentiment.model_json_schema(), indent=2)}",
                },
                {
                    "role": "user",
                    "content": f"Perform Sentiment Analysis on each the list of sentences and perform mean of all their values {sentences['Review'].tolist()}",
                },
            ],
            model="llama3-8b-8192",
            temperature=0,
            # Streaming is not supported in JSON mode
            stream=False,
            # Enable JSON mode by setting the response format
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content

        # Validate the JSON response against the Sentiment model
        sentiments = Sentiment.model_validate_json(response_content)
        return sentiments

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during sentiment analysis: {str(e)}")

@app.post("/sentiment_analyzer/")
async def sentiment_analyzer(file: UploadFile = File(...)):
    """
    Endpoint to perform sentiment analysis on uploaded CSV or Excel files.

    Args:
        file (UploadFile): The uploaded file containing reviews.

    Returns:
        List[Sentiment]: A list of Sentiment objects with analysis results.
    """
    try:
        # Read file content based on its type
        if file.content_type == 'text/csv':
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        elif file.content_type in [
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel'
        ]:
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail='Invalid file received. Please upload CSV or Excel files.')

        # Validate the DataFrame for expected columns
        if 'Review' not in df.columns:
            raise HTTPException(status_code=400, detail='Missing "Review" column in the uploaded file.')

        # Perform sentiment analysis
        return await get_sentiments(df)

    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail='The uploaded file is empty.')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task:app", host="127.0.0.1", port=8000, reload=True)
