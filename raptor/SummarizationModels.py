import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from langsmith.wrappers import wrap_openai

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


# noinspection PyArgumentList
class GPT4OSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-4o-mini", temperature=0.0):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        prompt = (
            "Write a summary of the following, including as many key details as possible::\n\n" + context +"\n" 
            f"Try to keep it below {int(max_tokens)} tokens."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=self.temperature,
            # langsmith_extra={
            #     "tags": ["summarization", "content-processing"],  # Add searchable tags
            #     "name": "Document Summarization",  # Custom name for the run
            #     "project_name": "lightbox-dev",  # Override project
            #     "metadata": {  # Custom metadata
            #         "document_length": len(context),
            #         "source_type": "article",
            #         "max_tokens_allowed": max_tokens
            #     },
            #     "raptor_step": "summarize"  # Your custom field
            # }
        )
        summary = response.choices[0].message.content.strip()
        return summary