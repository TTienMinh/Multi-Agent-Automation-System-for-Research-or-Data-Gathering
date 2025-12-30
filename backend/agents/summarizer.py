"""
Summarizer Agent Module
-----------------------
Provides a configurable agent for text summarization using Hugging Face models and LangChain integration.
"""
import torch
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummarizerConfig(BaseModel):
    """
    Configuration for the SummarizerAgent.

    Attributes:
        model_id (str): Hugging Face model name for summarization.
        device_map (str): Device mapping for model loading.
        max_new_tokens (int): Maximum number of tokens in the summary.
        temperature (float): Sampling temperature for generation.
        top_p (float): Nucleus sampling probability.
        repetition_penalty (float): Penalty for repeated phrases.
        do_sample (bool): Whether to use sampling.
        trust_remote_code (bool): Trust remote code from Hugging Face.
        load_in_4bit (bool): Load model in 4-bit precision for large models.
    """
    model_id: str = Field(default="Falconsai/text_summarization", description="Hugging Face model name for summarization")
    device_map: str = "auto"
    max_new_tokens: int = Field(512, ge=1, description="Max response length")
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    do_sample: bool = True
    trust_remote_code: bool = True
    load_in_4bit: bool = False  # Critical for running large models on consumer GPUs

class SummarizerAgent:
    """
    Agent for text summarization using Hugging Face models.

    Methods:
        __init__(config): Initialize the agent and load the model.
        _combine_chunks(chunks): Combine multiple text chunks into a single string.
        summarize(chunks): Summarize the combined text chunks.
    """
    def __init__(self, config: SummarizerConfig):
        """
        Initialize the SummarizerAgent with a specified Hugging Face model.

        Args:
            config (SummarizerConfig): Configuration for the summarizer agent.
        Raises:
            Exception: If model or tokenizer loading fails.
        """
        self.config = config
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_id,
                load_in_4bit=self.config.load_in_4bit,
                offload_folder="offload",
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Tokenizer and model for '{self.config.model_id}' initialized on device: {'cuda' if torch.cuda.is_available() else 'cpu'}.")
            logger.info(f"Model '{self.config.model_id}' loaded successfully.")
        except Exception as e:
            logger.exception(f"Error loading model '{self.config.model_id}': {e}")
            raise e

    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Combine text chunks into a single string for summarization.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dicts with 'page_content'.
        Returns:
            str: Combined text from all chunks.
        """
        if not chunks:
            logger.warning("No chunks provided to _combine_chunks.")
            return ""
        texts = [
            chunk.get('page_content', '') if isinstance(chunk, dict) else getattr(chunk, 'page_content', '') 
            for chunk in chunks
        ]
        logger.debug(f"Combined {len(texts)} chunks into a single text of length {sum(len(t) for t in texts)}.")
        return " ".join(texts)

    def summarize(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Summarizes the input chunks using the loaded model.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dicts with 'page_content'.
        Returns:
            str: Generated summary text.
        """
        try:
            combined_text = self._combine_chunks(chunks)
            if not combined_text:
                logger.warning("No content to summarize.")
                return ""
            logger.info(f"Starting summarization for text of length {len(combined_text)}.")
            
            summarization_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                trust_remote_code=self.config.trust_remote_code,
                do_sample=self.config.do_sample,
            )
            
            response = HuggingFacePipeline(pipeline=summarization_pipeline)
            summary = response.invoke(combined_text)
            logger.info(f"Summarization completed. Summary length: {len(summary)}.")
            return summary
        except Exception as e:
            logger.exception(f"Error during summarization: {e}")
            return ""


if __name__ == "__main__":
    # Define settings
    config = SummarizerConfig(
        model_id="facebook/bart-large-cnn",
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=False,
        trust_remote_code=True,
        load_in_4bit=False,
    )

    # Initialize Agent (Model loads NOW, only once)
    agent = SummarizerAgent(config)

    # Mock Data (Simulating LangChain documents)
    text = """ 
    World War II stands as the most widespread and deadliest conflict in human history, a cataclysmic event that engulfed the globe from 1939 to 1945 and fundamentally redrew the geopolitical map of the twentieth century. It was a struggle of ideologies pitting the Allied powers against the totalitarian Axis regimes, characterized by the terrifying advent of "total war" where the distinction between combatant and civilian was ruthlessly erased. The conflict mobilized the economic, industrial, and scientific capabilities of major nations to an unprecedented degree, transforming entire societies into engines of destruction and introducing industrialized slaughter on a scale previously unimaginable. From the frozen steppes of the Eastern Front and the hedgerows of Normandy to the sweltering jungles of the Pacific and the scorching sands of North Africa, the war was fought across land, sea, and air, driven by rapid technological advancements in aviation, rocketry, and armored warfare. The skies over Europe and Asia turned black with bombers flattening historic cities, while the oceans churned with the hidden menace of submarine warfare and the clash of massive naval fleets. Yet, beyond the military strategies lay a profound humanitarian nightmare; the systematic genocide of the Holocaust and the brutal occupation of sovereign nations revealed the darkest depths of human cruelty, leaving a scar on the collective conscience that would never fully heal. The conflict reached its harrowing crescendo with the dawn of the nuclear age, forever altering the nature of power and international relations. When the dust finally settled, leaving tens of millions dead and continents in ruins, the old colonial world order had been dismantled, paving the way for the Cold War, the formation of the United Nations, and a fragile, enduring hope that humanity might resolve future disputes without resorting to such apocalyptic devastation.
    """
    
    docs = [
        {"page_content": text},
    ]

    # Run
    summary = agent.summarize(docs)
    print(f"\nSummary: {summary}")