import openai
import pandas as pd
import json

class RAPTOREvaluator:
    def __init__(self, qa_pairs, ra_instance, model="gpt-4o-mini", temperature=0.0):
        """
        Initializes the evaluator.

        Parameters:
        - qa_pairs: A list of dictionaries, each with keys "question" and "example_answer".
        - ra_instance: An instance of RAPTOR (RetrievalAugmentation) used to retrieve answers.
        - model: The OpenAI model to use for evaluation (default "gpt-4o-mini").
        - temperature: The temperature for the API call (default 0.0 for deterministic output).
        """
        self.qa_pairs = qa_pairs
        self.ra = ra_instance
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI()

    def evaluate(self):
        """
        For each question/example pair, retrieves the answer using RAPTOR, prepares an evaluation prompt 
        that incorporates the example good answer, and queries the OpenAI API for evaluation metrics.
        
        Returns:
        - A pandas DataFrame with the following columns:
            "question", "example_answer", "retrieved_answer", "accuracy", 
            "completeness", "faithfulness", "relevance", and "feedback".
        """
        qa_results = []
        for qa in self.qa_pairs:
            question = qa["question"]
            example_answer = qa["example_answer"]
            # Query RAPTOR to retrieve an answer for the question.
            retrieved_answer = self.ra.answer_question(question=question)

            # Prepare the evaluation prompt that includes the reference example answer.
            prompt = f"""
            Evaluate the following Question & Answer pair based on Medicare guidelines. 
            Consider the provided example good answer as the reference standard.
            
            Evaluation Metrics (0-10):
            - Accuracy: Does the retrieved answer correctly address the question based on Medicare's official guidelines?
            - Completeness: Does the retrieved answer include all relevant details compared to the example answer?
            - Faithfulness: Does the retrieved answer strictly use information from the document without introducing hallucinations?
            - Relevance: Does the retrieved answer stay focused on the question without including unrelated or extraneous information?
            
            Please respond ONLY with a JSON object in the format:
            {{"feedback": str, "metrics": {{"accuracy": int, "completeness": int, "faithfulness": int, "relevance": int}}}}
            
            Do NOT include markdown, code blocks, or additional explanations. We will directly parse your output.
            
            ---
            Question: {question}
            Example Good Answer: {example_answer}
            Retrieved Answer: {retrieved_answer}
            """

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of Medicare compliance documents. Return a JSON object only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature
                )
                evaluation_text = response.choices[0].message.content.strip()

                # Parse the JSON response
                evaluation_data = json.loads(evaluation_text)
                if "metrics" in evaluation_data and all(
                    key in evaluation_data["metrics"] for key in ["accuracy", "completeness", "faithfulness", "relevance"]
                ):
                    qa_result = {
                        "question": question,
                        "example_answer": example_answer,
                        "retrieved_answer": retrieved_answer,
                        "accuracy": evaluation_data["metrics"]["accuracy"],
                        "completeness": evaluation_data["metrics"]["completeness"],
                        "faithfulness": evaluation_data["metrics"]["faithfulness"],
                        "relevance": evaluation_data["metrics"]["relevance"],
                        "feedback": evaluation_data["feedback"]
                    }
                else:
                    raise ValueError("Missing expected JSON keys in response.")
            except Exception as e:
                qa_result = {
                    "question": question,
                    "example_answer": example_answer,
                    "retrieved_answer": retrieved_answer,
                    "accuracy": None,
                    "completeness": None,
                    "faithfulness": None,
                    "relevance": None,
                    "feedback": f"Error parsing evaluation: {str(e)}"
                }
            qa_results.append(qa_result)

        return pd.DataFrame(qa_results)