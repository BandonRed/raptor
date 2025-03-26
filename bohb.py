import csv
import json
import os
import openai
import fitz
from langsmith import Client
from typing import Dict, Any
from raptor.RetrievalAugmentation import RetrievalAugmentation, RetrievalAugmentationConfig

# Ray Tune and BOHB imports
import ray
from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.tune.search import ConcurrencyLimiter
from ray.tune.callback import Callback

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


import pandas as pd
import tempfile
import time
from pathlib import Path

plt.ion()  # Enable interactive plotting

# Base hyperparameters (fixed)
BASE_HYPERPARAMETERS = {
    "tb_max_tokens": 500,
    "tb_num_layers": 5,
    "tr_threshold": 0.75,
    "tb_threshold": 0.75,
    "tb_top_k": 5,
    "tr_top_k": 5,
    "qa_model": None,
    "embedding_model": None,
    "summarization_model": None,
}

# Define hyperparameter search space as a dict.
# Used for finding the best parameter set
search_space = {
    "tb_threshold": tune.choice([0.5, 0.75]),
    "tb_top_k": tune.choice([3, 5]),
    "tb_summarization_length": tune.choice([125, 250, 500]),
}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n\n".join(page.get_text("text") for page in doc)
    return text

def init_raptor(config: Dict[str, Any]) -> RetrievalAugmentation:
    # Ensure tb_top_k is an integer and at least 1.
    if "tb_summarization_length" in config:
        config["tb_summarization_length"] = int(config["tb_summarization_length"])
    if "tb_top_k" in config:
        config["tb_top_k"] = int(config["tb_top_k"])
        if config["tb_top_k"] < 1:
            raise ValueError("tb_top_k must be at least 1")
    
    merged_params = BASE_HYPERPARAMETERS.copy()
    merged_params.update(config)
    
    custom_config = RetrievalAugmentationConfig(**merged_params)
    ra = RetrievalAugmentation(config=custom_config)
    
    pdf_path = "/Users/awick/venvs/lightbox/raptor/raptor/pdfs/clm104c13.pdf"
    pdf_text = extract_text_from_pdf(pdf_path)
    ra.add_documents(pdf_text[5820:166840])
    return ra

# Global RAPTOR instance.
RAPTOR_INSTANCE = None

def raptor_target(inputs: Dict[str, Any]) -> Dict[str, Any]:
    question = inputs.get("Question", "")
    if not question:
        return {"output": "No question provided"}
    answer = RAPTOR_INSTANCE.answer_question(question=question)
    return {"output": answer}

def raptor_evaluator(run_outputs, reference_outputs):
    client = openai.OpenAI()
    model = "gpt-4o-mini"
    temperature = 0.0

    try:
        question = ""
        if hasattr(run_outputs, "inputs") and isinstance(run_outputs.inputs, dict):
            question_data = run_outputs.inputs.get("inputs", {})
            if isinstance(question_data, dict):
                question = question_data.get("Question", "")
        
        retrieved_answer = ""
        if hasattr(run_outputs, "outputs") and isinstance(run_outputs.outputs, dict):
            retrieved_answer = run_outputs.outputs.get("output", "")
        
        example_answer = ""
        if hasattr(reference_outputs, "outputs") and isinstance(reference_outputs.outputs, dict):
            example_answer = reference_outputs.outputs.get("message", "")
        
        prompt = f"""
Evaluate the following Question & Answer pair based on Medicare guidelines.
Consider the provided example good answer as the reference standard.

Evaluation Metrics (0-10):
- Accuracy: Does the retrieved answer correctly address the question?
- Completeness: Does it include all relevant details compared to the example answer?
- Faithfulness: Does it strictly use information from the document?
- Relevance: Does it stay focused on the question?

Respond ONLY with a JSON object:
{{"feedback": str, "metrics": {{"accuracy": int, "completeness": int, "faithfulness": int, "relevance": int}}}}

---
Question: {question}
Example Good Answer: {example_answer}
Retrieved Answer: {retrieved_answer}
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator of Medicare compliance documents. Return a JSON object only, without triple backticks or code blocks."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        evaluation_text = response.choices[0].message.content.strip()
        try:
            evaluation_data = json.loads(evaluation_text)
        except Exception as e:
            evaluation_text = evaluation_text.replace("```json", "").replace("```", "").strip()
            evaluation_data = json.loads(evaluation_text)
        metrics = evaluation_data["metrics"]
        overall_score = (metrics["accuracy"] + metrics["completeness"] +
                         metrics["faithfulness"] + metrics["relevance"]) / 40.0
        return {
            "score": overall_score,
            "accuracy": metrics["accuracy"] / 10.0,
            "completeness": metrics["completeness"] / 10.0,
            "faithfulness": metrics["faithfulness"] / 10.0,
            "relevance": metrics["relevance"] / 10.0,
            "feedback": evaluation_data["feedback"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "score": 0.0,
            "accuracy": 0.0,
            "completeness": 0.0,
            "faithfulness": 0.0,
            "relevance": 0.0,
            "feedback": f"Error during evaluation: {str(e)}"
        }

class PlottingCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        data = []
        for t in trials:
            if t.last_result is not None:
                config = t.config
                data.append({
                    "tb_threshold": config.get("tb_threshold"),
                    "tb_top_k": config.get("tb_top_k"),
                    "tb_summarization_length": config.get("tb_summarization_length"),
                    "score": t.last_result.get("score"),
                })
        if not data:
            return
        df = pd.DataFrame(data)
        plt.clf()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].scatter(df["tb_threshold"], df["score"], c="blue", alpha=0.7)
        axes[0].set_title("tb_threshold vs Score")
        axes[0].set_xlabel("tb_threshold")
        axes[0].set_ylabel("Score")
        axes[1].scatter(df["tb_top_k"], df["score"], c="green", alpha=0.7)
        axes[1].set_title("tb_top_k vs Score")
        axes[1].set_xlabel("tb_top_k")
        axes[1].set_ylabel("Score")
        axes[2].scatter(df["tb_summarization_length"], df["score"], c="red", alpha=0.7)
        axes[2].set_title("tb_summarization_length vs Score")
        axes[2].set_xlabel("tb_summarization_length")
        axes[2].set_ylabel("Score")
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        plt.savefig("trial_plot.png")  # Save the current figure to a file
        
def run_evaluation(config):
    """
    Objective function for Ray Tune using BOHB.
    Initializes RAPTOR with the given config, runs evaluation,
    aggregates the results, and reports an average score.
    """
    client = Client()
    dataset_name = "rules-engine-qa"
    
    global RAPTOR_INSTANCE
    RAPTOR_INSTANCE = init_raptor(config)
    
    experiment_results = client.evaluate(
        raptor_target,
        data=dataset_name,
        evaluators=[raptor_evaluator],
        experiment_prefix="raptor-medicare-qa-bohb-new-prompt-500",
        metadata={"hyperparameters": config, "RAG method": "RAPTOR"},
        max_concurrency=2
    )
    
    total_score = sum(r['evaluation_results']['results'][0].score for r in experiment_results._results)
    avg_score = round(float(total_score) / len(experiment_results._results), 4)
    assert isinstance(avg_score, float), "score wrong dtype"
    assert 0 < avg_score < 1, "score out of expected range (0,1)"
    print(f"Reporting aggregated score: {avg_score} for config: {config}")
    tune.report(metrics={"score": avg_score})

if __name__ == "__main__":
    ray.shutdown()
    ray.init(local_mode=False)
    
    
    bohb_search = TuneBOHB(metric="score", mode="max")  # Instantiate the BOHB search algorithm.
    bohb_search = ConcurrencyLimiter(bohb_search, max_concurrent=7)
    bohb_scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="score",
        mode="max",
        max_t=100,
        reduction_factor=3
    )
    
    # For local storage, use an absolute path with file:// prefix.
    storage_dir = os.path.abspath("bohb_results")
    storage_path = f"file://{storage_dir}"
    tuner = tune.Tuner(
        run_evaluation,
        tune_config=tune.TuneConfig(
            search_alg=bohb_search,
            scheduler=bohb_scheduler,
            num_samples=8,
        ),
        run_config=tune.RunConfig(
            name="bohb_exp",
            stop={"training_iteration": 100},
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    
    best_trial = results.get_best_result(metric="score", mode="max")
    if best_trial is not None:
        best_config = best_trial.config
        best_score = best_trial.metrics["score"]
        csv_file = "hyperparam_best_results_weds.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tb_threshold", "tb_top_k", "tb_summarization_length", "score"])
            writer.writeheader()
            row = best_config.copy()
            row["score"] = best_score
            writer.writerow(row)
        print("BOHB search complete. Best config and score saved to", csv_file)
    else:
        print("No best trial found. Check your experiment results.")