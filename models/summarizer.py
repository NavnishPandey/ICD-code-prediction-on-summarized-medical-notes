import ray
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
import logging
logging.basicConfig(level=logging.INFO)

@ray.remote(num_gpus=1)
class Summarizer:
    def __init__(self, model_name="google/gemma-2b-it"):
        try:
            logging.info("Loading tokenizer and model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

            hf_pipeline = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                return_full_text=False
            )

            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

            self.prompt_template = PromptTemplate(
                input_variables=["dialogue"],
                template=(
                    "You are a helpful assistant. Summarize the following medical conversation "
                    "between a doctor and a patient in a concise and clear note.\n\n"
                    "{dialogue}\n\nSummary:"
                )
            )

            self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
            logging.info("Summarizer initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Summarizer: {e}")
            raise

    def summarize(self, dialogue: str) -> str:
        logging.info("Summarizing dialogue...")
        try:
            result = self.chain.run({"dialogue": dialogue})
            return result.strip()
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            raise
