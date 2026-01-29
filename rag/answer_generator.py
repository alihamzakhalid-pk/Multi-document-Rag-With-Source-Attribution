"""
Answer Generator for the Multi-Document RAG System.

Handles LLM-based answer generation with strict grounding to retrieved
context and explicit source attribution. Implements anti-hallucination
policies and returns structured JSON responses.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from groq import Groq

from config.settings import get_settings
from utils.logger import get_logger
from .retriever import RetrievedChunk

logger = get_logger(__name__)


# The STRICT system prompt that enforces RAG behavior
RAG_SYSTEM_PROMPT = """You are a **Retrieval-Augmented Generation (RAG) answering engine** designed for a **production-grade multi-document QA system**.

Your behavior is governed by the following **strict and non-negotiable rules**:

---

### **CORE ROLE**

Your task is to **answer user questions using ONLY the retrieved document chunks provided to you**.
You must **never use prior knowledge, assumptions, or external information**.

You operate in a **grounded, retrieval-first mode**.

---

### **MANDATORY ANSWERING RULES (CRITICAL)**

1. **Use ONLY retrieved context**
   * Do not add, infer, guess, or complete missing information.
   * If the answer is not explicitly stated in the retrieved chunks, you must refuse.

2. **Strict anti-hallucination policy**
   * If the retrieved context does NOT contain the answer, respond EXACTLY with:
     "The provided documents do not contain this information."

3. **Explicit source attribution**
   * Every factual statement in your answer **must map to at least one retrieved chunk**.
   * You may only cite sources that appear in the retrieved context.

4. **Multi-document awareness**
   * You may combine information from multiple documents.
   * You must list **all documents used** in the final sources list.

---

### **RESPONSE FORMAT (STRICT — NO DEVIATIONS)**

You MUST return a JSON object in the following exact format:

```json
{
  "answer": "<concise, factual answer strictly based on retrieved context>",
  "sources": [
    {
      "document_name": "<name from metadata>",
      "page": <page_number>,
      "chunk_id": "<chunk_id>"
    }
  ]
}
```

#### Formatting rules:
* NO markdown outside JSON
* NO explanations outside JSON
* NO missing fields
* NO empty sources array if an answer is provided
* NO sources that were not in retrieved context

---

### **WHEN INFORMATION IS INSUFFICIENT**

If the retrieved chunks do **not** contain the answer:

```json
{
  "answer": "The provided documents do not contain this information.",
  "sources": []
}
```

This behavior is **mandatory**.

---

### **PROHIBITED BEHAVIOR**

- Using world knowledge
- Guessing or completing partial facts
- Answering without sources
- Citing sources not in retrieved chunks
- Explaining your reasoning process
- Deviating from the response schema

---

### **MENTAL MODEL**

Think like a **legal document analyst**:
* If it's written → you may answer
* If it's not written → you must refuse

Return ONLY the JSON response, nothing else."""


@dataclass
class SourceReference:
    """Reference to a source document chunk."""
    document_name: str
    page: int
    chunk_id: str
    is_section: bool = False  # True for DOCX sections, False for PDF pages
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "is_section": self.is_section
        }


@dataclass
class AnswerResponse:
    """Structured response from the answer generator."""
    answer: str
    sources: List[SourceReference]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources]
        }


class AnswerGenerator:
    """
    Generates answers from retrieved context using an LLM.
    
    Enforces strict grounding to retrieved chunks and returns
    structured responses with source attribution.
    """
    
    NO_INFO_RESPONSE = "The provided documents do not contain this information."
    
    def __init__(self):
        """Initialize the answer generator with the configured LLM."""
        settings = get_settings()
        
        if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY not configured. Answer generation will fail.")
            self.client = None
        else:
            self.client = Groq(api_key=settings.groq_api_key)
        
        self.model = settings.llm_model
        logger.info(f"AnswerGenerator initialized with model: {self.model}")
    
    def generate(
        self, 
        question: str, 
        retrieved_chunks: List[RetrievedChunk]
    ) -> AnswerResponse:
        """
        Generate an answer based on retrieved chunks.
        
        Args:
            question: The user's question.
            retrieved_chunks: Chunks retrieved from the vector store.
        
        Returns:
            AnswerResponse: Structured answer with sources.
        """
        if not self.client:
            raise RuntimeError(
                "GROQ_API_KEY not configured. Please set it in your .env file."
            )
        
        # Handle empty context
        if not retrieved_chunks:
            logger.info("No chunks retrieved, returning no-info response")
            return AnswerResponse(
                answer=self.NO_INFO_RESPONSE,
                sources=[]
            )
        
        # Format context for the LLM
        context = self._format_context(retrieved_chunks)
        
        # Build the user message
        user_message = self._build_user_message(question, context)
        
        logger.info(f"Generating answer for: {question[:50]}...")
        
        try:
            # Call the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,  # Deterministic for consistency
                max_tokens=1024,
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse the JSON response
            return self._parse_response(response_text, retrieved_chunks)
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """
        Format retrieved chunks as context for the LLM.
        
        Args:
            chunks: Retrieved chunks with metadata.
        
        Returns:
            str: Formatted context string.
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"--- Retrieved Chunk {i} ---\n"
                f"document_name: {chunk.document_name}\n"
                f"page_number: {chunk.page_number}\n"
                f"chunk_id: {chunk.chunk_id}\n"
                f"text: {chunk.text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _build_user_message(self, question: str, context: str) -> str:
        """
        Build the user message with question and context.
        
        Args:
            question: User's question.
            context: Formatted context string.
        
        Returns:
            str: Complete user message.
        """
        return f"""### User Question
{question}

### Retrieved Document Chunks
{context}

### Instructions
Based ONLY on the retrieved document chunks above, answer the user's question.
Return your response as a JSON object with "answer" and "sources" fields.
If the answer is not in the chunks, respond with the no-information message."""
    
    def _parse_response(
        self, 
        response_text: str, 
        retrieved_chunks: List[RetrievedChunk]
    ) -> AnswerResponse:
        """
        Parse the LLM response into a structured AnswerResponse.
        
        Args:
            response_text: Raw response from the LLM.
            retrieved_chunks: Original retrieved chunks for validation.
        
        Returns:
            AnswerResponse: Parsed and validated response.
        """
        # Try to extract JSON from the response
        try:
            # Remove markdown code blocks if present
            json_text = response_text
            if "```json" in json_text:
                json_text = re.search(r'```json\s*(.*?)\s*```', json_text, re.DOTALL)
                if json_text:
                    json_text = json_text.group(1)
                else:
                    json_text = response_text
            elif "```" in json_text:
                json_text = re.search(r'```\s*(.*?)\s*```', json_text, re.DOTALL)
                if json_text:
                    json_text = json_text.group(1)
                else:
                    json_text = response_text
            
            # Parse JSON
            parsed = json.loads(json_text.strip())
            
            # Extract answer
            answer = parsed.get("answer", self.NO_INFO_RESPONSE)
            
            # Extract and validate sources
            raw_sources = parsed.get("sources", [])
            # Build lookup for is_section by chunk_id
            chunk_section_map = {chunk.chunk_id: getattr(chunk, 'is_section', False) for chunk in retrieved_chunks}
            valid_chunk_ids = set(chunk_section_map.keys())
            
            sources = []
            for source in raw_sources:
                chunk_id = source.get("chunk_id", "")
                # Validate that the source was in retrieved context
                if chunk_id in valid_chunk_ids:
                    sources.append(SourceReference(
                        document_name=source.get("document_name", "unknown"),
                        page=source.get("page", 0),
                        chunk_id=chunk_id,
                        is_section=chunk_section_map.get(chunk_id, False)
                    ))
                else:
                    logger.warning(f"Filtered out invalid source: {chunk_id}")
            
            # If answer is provided but no valid sources, this is suspicious
            if answer != self.NO_INFO_RESPONSE and not sources:
                logger.warning("Answer provided without valid sources, checking chunks")
                # Try to add sources from retrieved chunks that might be relevant
                # This is a safety measure but keeps strict grounding
                pass  # Keep empty sources to flag potential issue
            
            return AnswerResponse(answer=answer, sources=sources)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Return a safe fallback
            return AnswerResponse(
                answer=self.NO_INFO_RESPONSE,
                sources=[]
            )
    
    def generate_sync(
        self, 
        question: str, 
        retrieved_chunks: List[RetrievedChunk]
    ) -> Dict[str, Any]:
        """
        Generate an answer and return as dictionary.
        
        Convenience method that returns a dict instead of AnswerResponse.
        
        Args:
            question: The user's question.
            retrieved_chunks: Chunks retrieved from the vector store.
        
        Returns:
            Dict: Answer response as dictionary.
        """
        response = self.generate(question, retrieved_chunks)
        return response.to_dict()
