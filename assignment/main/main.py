#main.py
from fastapi import FastAPI, HTTPException
import asyncio
from transformers import pipeline
import random
from main.logger import logger  # Ensure logger.py is correctly configured
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle 

# Check if CUDA is available
import torch
device = 0 if torch.cuda.is_available() else -1

serve.start(detached=True, http_options={"host": "0.0.0.0"})

#app = FastAPI()
ray.init(ignore_reinit_error=True,log_to_driver=True)
logger.info("Starting FastAPI app")


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
#@serve.deployment
# Classes for the ML pipeline
class BARTsummarizer:
    def __init__(self):
        logger.info("Summarizer initialized with BART model")
        self.model = pipeline("summarization", model="facebook/bart-large-cnn",device=device)
        print("BERT INTialized")
        

    def summarize(self, text: str) -> str:
        try:
            print("text:"+text)
            model_output = self.model(text)
            summary = model_output[0]["summary_text"]
            logger.info(f"Summarized text: {summary}")
            print(f"Summarized text: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            raise

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
#@serve.deployment
class SentimentAnalyzer:
    def __init__(self):
        self.roberta = pipeline("sentiment-analysis", model="roberta-base",device=device)
        self.distilbert = pipeline("sentiment-analysis", model="distilbert-base-uncased",device=device)
        self.label_mapping = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
        logger.info("SentimentAnalyzer initialized with RoBERTa and DistilBERT models")
        print("SentimentAnalyzer Intialized")

    async def analyze_sentiment(self, text: str) -> dict:
        coin_toss = random.choice(["RoBERTa", "DistilBERT"])
        try:
            if coin_toss == "RoBERTa":
                result = self.roberta(text)
                model = "RoBERTa"
            else:
                result = self.distilbert(text)
                model = "DistilBERT"

            sentiment = self.label_mapping.get(result[0]["label"], "UNKNOWN")
            confidence = result[0]["score"]
            
            logger.info(f"Sentiment analysis: Model={model}, Sentiment={sentiment}, Confidence={confidence}")
            return {"Model": model, "Sentiment": sentiment, "Confidence": confidence}
        except Exception as e:
            logger.error(f"Error during sentiment analysis: {e}")
            raise

@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2, "num_gpus": 0})
#@serve.deployment
class LLMResponder:
    def __init__(self):
        self.model = pipeline("text-generation", model="gpt2",device=device)
        logger.info("LLMResponder initialized with GPT-2 model")
        print("LLMResponder Intialized")

    async def generate_response(self, input_text: str) -> dict:
        try:
            response = self.model(
                input_text,
                max_length=500,
                truncation=True,
                pad_token_id=self.model.tokenizer.eos_token_id,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            generated_text = response[0]["generated_text"]
            logger.info(f"Generated LLM response: {generated_text}")
            return {"response": generated_text}
        except Exception as e:
            logger.error(f"Error during LLM response generation: {e}")
            raise


@serve.deployment
# @serve.ingress(app)
class endpoint:
    def __init__( self, BARTsummarizer: DeploymentHandle,SentimentAnalyzer: DeploymentHandle,LLMResponder: DeploymentHandle):

        self.BARTsummarizer = BARTsummarizer
        self.SentimentAnalyzer = SentimentAnalyzer
        self.LLMResponder = LLMResponder

    
    # Pipeline orchestration function
    async def process_text_pipeline(self, message: str):
        try:
            logger.info("Starting text processing pipeline")
            print(f"Input text: {message}")
            
            # Step 1: Summarize the input text
            summary_ref = self.BARTsummarizer.summarize.remote(message)
            summary_text = await summary_ref  # Wait for the summarization to complete
            print(f"Summary: {summary_text}")

            # Step 2: Analyze sentiment of the summary
            sentiment_task = self.SentimentAnalyzer.analyze_sentiment.remote(summary_text)
            print("Sentiment task initiated.")

            # Step 3: Generate LLM response based on the summary
            llm_response_task = self.LLMResponder.generate_response.remote(summary_text)
            print("LLM response task initiated.")

            # Gather sentiment analysis and LLM response concurrently
            sentiments, llm_response = await asyncio.gather(sentiment_task, llm_response_task)
            logger.info(f"Pipeline results: Summary={summary_text}, Sentiments={sentiments}, LLM Response={llm_response}")
            
            return summary_text, sentiments, llm_response

        except Exception as e:
            logger.error(f"Error in text processing pipeline: {e}")
            raise

    
    # API endpoint
    #@app.post("/process/")
    async def __call__(self,http_request):
        request=await http_request.json()
        print("request:\n"+request.get('text'))
        try:
            logger.info(f"Received input text: {request.get('text')}")
            summary, sentiments, response = await self.process_text_pipeline(request.get('text'))
            logger.info("Successfully processed text")
            return {
                "summary": summary,
                "sentiments": sentiments,
                "llm_response": response
            }
        except Exception as e:
            logger.error(f"Error in /process/ endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))


summarizer = BARTsummarizer.bind()
sentiment_analyzer = SentimentAnalyzer.bind()
llm_responder = LLMResponder.bind()

endpoint_app=endpoint.bind(summarizer,sentiment_analyzer,llm_responder)

#serve.run(endpoint_app)



