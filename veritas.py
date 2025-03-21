import openai
import json
import os
from typing import Optional, List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd 
# Define default models at the top of the file
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o"

try:
    # Try importing the new version first
    from google import genai
    GEMINI_NEW_API = True
    GEMINI_AVAILABLE = True
except ImportError:
    # Fall back to old version
    try:
        import google.generativeai as genai
        GEMINI_NEW_API = False
        GEMINI_AVAILABLE = True
    except ImportError:
        GEMINI_AVAILABLE = False

class BaseLLMClient:
    """Base class for LLM clients with common functionality"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_OPENAI_MODEL):
        """Initialize base LLM client
        
        Args:
            api_key (str, optional): API key for the provider
            model (str): Default model to use
        """
        self.api_key = api_key
        self.model = model
        self._client = None
        self.is_initialized = False
    
    @property
    def client(self):
        """Return the underlying client instance, initializing if needed"""
        if not self.is_initialized:
            self._initialize_client()
        return self._client
    
    def _initialize_client(self):
        """Initialize the underlying client - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _initialize_client")
    
    def generate_completion(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using the LLM client"""
        raise NotImplementedError("Subclasses must implement generate_completion")
    
    def get_provider(self) -> str:
        """Return the provider name (e.g., 'openai', 'gemini')"""
        raise NotImplementedError("Subclasses must implement get_provider")

class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_OPENAI_MODEL):
        super().__init__(api_key, model)
    
    def _initialize_client(self):
        """Initialize the OpenAI client with API key."""
        self._client = openai.OpenAI(api_key=self.api_key)
        self.is_initialized = True
    
    def generate_completion(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using OpenAI"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            **kwargs
        )
        
        return response.choices[0].message.content.strip()
    
    def get_provider(self) -> str:
        return "openai"

class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models with much larger context windows"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GEMINI_MODEL, verbose: bool = False):
        super().__init__(api_key, model)
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI package not available. Install with: pip install google-generativeai")
        self.verbose = verbose
    
    def _initialize_client(self):
        """Initialize the Gemini client with API key."""
        api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or constructor.")
        
        if GEMINI_NEW_API:
            # New API style (google.genai)
            self._client = genai.Client(api_key=api_key)
            if self.verbose:
                print(f"[INFO] Initialized Gemini with new API (google.genai)")
        else:
            # Old API style (google.generativeai)
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(model_name=self.model)
            if self.verbose:
                print(f"[INFO] Initialized Gemini with legacy API (google.generativeai)")
            
        self.is_initialized = True
    
    def generate_completion(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using Gemini"""
        from google.genai import types
        
        # Handle system prompt by prepending to user prompt if needed
        if system_prompt:
            final_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            final_prompt = prompt
            
        
        # Set up generation parameters
        temp_val = temperature if temperature is not None else 0.0
        tokens_val = max_tokens or 8192
        model_name = model or self.model
        
        try:
            if GEMINI_NEW_API:
                # New API style - the parameters are passed directly, not in a generation_config object
                if self.verbose:
                    print(f"[INFO] Generating with new API: model={model_name}, max_tokens={tokens_val}, temp={temp_val}")
                
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=final_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=tokens_val,
                        temperature=temp_val,
                    ),
                )
                return response.text.strip()
            else:
                # Old API style
                generation_config = {
                    "max_output_tokens": tokens_val,
                    "temperature": temp_val,
                }
                
                if self.verbose:
                    print(f"[INFO] Generating with legacy API: model={model_name}, config={generation_config}")
                
                # Use provided model or default
                if model and model != self.model:
                    model_client = genai.GenerativeModel(model_name=model_name)
                else:
                    model_client = self.client
                    
                # Generate content
                response = model_client.generate_content(final_prompt, generation_config=generation_config)
                
                return response.text.strip()
        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Gemini generation failed: {str(e)}")
            raise
    
    def get_provider(self) -> str:
        return "gemini"

class IndependentKnowledgeSource:
    """
    Creates an independent knowledge source to avoid the chicken/egg problem
    in evaluation by establishing ground truth separate from the RAG system.
    Supports both OpenAI and Gemini models with automatic provider selection.
    """
    def __init__(self, document, model=DEFAULT_OPENAI_MODEL, client=None, use_gemini=False, verbose=False):
        self.document = document
        self.model = model
        self.verbose = verbose
        
        # Create appropriate client based on model or explicit flag
        if client:
            self.client = client
        elif use_gemini or 'gemini' in model.lower():
            if not GEMINI_AVAILABLE:
                raise ImportError("To use Gemini, install the package: pip install google-generativeai")
            self.client = GeminiClient(
                model=model if 'gemini' in model.lower() else DEFAULT_GEMINI_MODEL, 
                verbose=verbose
            )
        else:
            self.client = OpenAIClient(model=model)
            
        # Set chunking parameters based on model capabilities
        if use_gemini or 'gemini' in model.lower():
            # Gemini models can handle large contexts
            self.chunk_size = 60000  # ~60K chars, approximately 15K tokens
            self.chunk_overlap = 2000
        else:
            # GPT-4o has a 128K token limit (~400K chars)
            self.chunk_size = 8000
            self.chunk_overlap = 1000
            
        self.knowledge_chunks = self._create_knowledge_chunks()
        
        if self.verbose:
            print(f"[INFO] Document size: {len(self.document)} characters")
            print(f"[INFO] Split into {len(self.knowledge_chunks)} chunks")
            for i, chunk in enumerate(self.knowledge_chunks):
                print(f"[INFO] Chunk {i+1}/{len(self.knowledge_chunks)}: {len(chunk)} characters")
    
    def _clean_llm_output(self, text):
        """
        Clean and repair LLM-generated outputs using a stack-based approach
        for handling mismatched brackets and other JSON issues.
        """
        # First, clean up common JSON wrapper patterns
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            # Try standard parsing first
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"[WARNING] JSON parse error: {str(e)}, attempting to fix...")
            
            try:
                # Fix unterminated string issue first (common with LLMs)
                if cleaned_text.count('"') % 2 == 1:
                    # Find the last opening quote without a closing quote
                    last_idx = cleaned_text.rindex('"')
                    if last_idx > 0 and cleaned_text[last_idx-1] != '\\':  # Not an escaped quote
                        cleaned_text += '"'  # Add missing closing quote
                        if self.verbose:
                            print("[INFO] Added missing closing quote")
                
                # Stack-based approach to fix bracket mismatches
                stack = []
                valid_pairs = {'{': '}', '[': ']', '"': '"'}
                # We only track structural brackets, not quotes inside string literals
                in_string = False
                escaped = False
                
                # First pass: check for proper structure
                for i, char in enumerate(cleaned_text):
                    if char == '\\' and not escaped:
                        escaped = True
                        continue
                    
                    if char == '"' and not escaped:
                        in_string = not in_string
                    
                    if not in_string:  # Only process brackets outside of strings
                        if char in '{[':
                            stack.append(char)
                        elif char in '}]':
                            if not stack:  # Extra closing bracket
                                # This is an error, but we'll leave it for now
                                pass
                            else:
                                opening = stack.pop()
                                if valid_pairs[opening] != char:  # Mismatched brackets
                                    pass  # We'll handle this in the repair phase
                    
                    escaped = False if escaped else escaped
                
                # Add missing closing brackets based on stack content
                if stack:
                    # Add missing closing brackets in reverse order
                    closing_brackets = ''.join(valid_pairs[bracket] for bracket in reversed(stack))
                    fixed_text = cleaned_text + closing_brackets
                    if self.verbose:
                        print(f"[INFO] Adding missing closing brackets: {closing_brackets}")
                else:
                    fixed_text = cleaned_text
                
                # Try to parse the fixed JSON
                parsed = json.loads(fixed_text)
                if self.verbose:
                    print("[INFO] Successfully fixed JSON using stack-based approach")
                return parsed
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Stack-based repair failed: {str(e)}")
                
                try:
                    # Fallback: try more aggressive approaches
                    if self.verbose:
                        print("[INFO] Attempting more aggressive repair techniques...")
                    
                    # If we're dealing with an array, try to extract complete objects
                    if cleaned_text.strip().startswith('['):
                        import re
                        # Extract complete JSON objects with balanced braces
                        object_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
                        items = re.findall(object_pattern, cleaned_text)
                        if items:
                            repaired = '[' + ','.join(items) + ']'
                            result = json.loads(repaired)
                            if self.verbose:
                                print(f"[INFO] Extracted {len(items)} complete objects")
                            return result
                    
                    if self.verbose:
                        print("[ERROR] All JSON repair attempts failed")
                    return []  # Return empty list as fallback
                except Exception:
                    if self.verbose:
                        print("[ERROR] Failed to salvage any valid JSON")
                    return []  # Return empty list as fallback
        
    def _create_knowledge_chunks(self):
        """Split document based on model capabilities"""
        # For Gemini models with smaller documents, process entire document at once
        if 'gemini' in self.model.lower() and len(self.document) < 200000:
            if self.verbose:
                print(f"[INFO] Document under 200k chars, using Gemini to process entire document at once")
            return [self.document]
            
        # Otherwise, use smart splitting based on document structure
        if self.verbose:
            print(f"[INFO] Using smart document splitting (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
            
        # Look for natural document structure to guide chunking
        if "\n## " in self.document or "\n# " in self.document:
            # Document appears to have markdown headings - use these as natural boundaries
            if self.verbose:
                print(f"[INFO] Detected markdown headings, using them as chunk boundaries")
            
            # Custom chunk by heading implementation that respects maximum chunk size
            chunks = self._chunk_by_headings(self.document)
            
            if self.verbose:
                print(f"[INFO] Created {len(chunks)} chunks based on document structure")
            
            return chunks
        else:
            # No clear structure, fallback to LangChain's RecursiveCharacterTextSplitter
            if self.verbose:
                print(f"[INFO] No clear document structure detected, using recursive character splitting")
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            return text_splitter.split_text(self.document)
    
    def _chunk_by_headings(self, document):
        """Split document by markdown headings while respecting maximum chunk size"""
        import re
        
        # Define heading pattern - match # or ## at the start of a line
        heading_pattern = re.compile(r'^\s*#{1,2}\s+', re.MULTILINE)
        
        # Find all heading positions
        heading_matches = list(heading_pattern.finditer(document))
        heading_positions = [match.start() for match in heading_matches]
        
        # Add document start and end as boundary positions
        all_positions = [0] + heading_positions + [len(document)]
        
        chunks = []
        current_chunk_start = 0
        current_chunk_text = ""
        
        for i in range(1, len(all_positions)):
            section_text = document[all_positions[i-1]:all_positions[i]]
            
            # If adding this section would exceed max chunk size, start a new chunk
            if len(current_chunk_text) + len(section_text) > self.chunk_size and len(current_chunk_text) > 0:
                chunks.append(current_chunk_text)
                current_chunk_start = all_positions[i-1]
                current_chunk_text = section_text
            else:
                current_chunk_text += section_text
        
        # Add the final chunk if it has content
        if current_chunk_text:
            chunks.append(current_chunk_text)
            
        return chunks
    
    def extract_key_facts(self, max_facts=None):
        """Extract structured knowledge from document as ground truth"""
        all_facts = []
        system_prompt = "You are a Medicare documentation expert creating a structured knowledge base."
        
        if self.verbose:
            print(f"[INFO] Beginning fact extraction from {len(self.knowledge_chunks)} chunks")
        
        for i, chunk in enumerate(self.knowledge_chunks):
            if self.verbose:
                print(f"[INFO] Processing chunk {i+1}/{len(self.knowledge_chunks)} ({len(chunk)} chars)")
            
            prompt = f"""
            Extract key Medicare documentation facts from this text as structured knowledge.
            For each fact:
            1. State the requirement/rule precisely
            2. Cite specific documentation elements required
            3. Note any exceptions or special conditions
            
            Format as a JSON array of objects with fields:
            - "topic": general category (e.g., "MRI Documentation", "ICD Coding")
            - "requirement": specific requirement or rule
            - "documentation_elements": array of required elements
            - "exceptions": any exceptions to the rule
            
            Document chunk {i+1}/{len(self.knowledge_chunks)}:
            {chunk}
            """
            
            try:
                if self.verbose:
                    print(f"[INFO] Sending chunk {i+1} to {self.client.get_provider()} model {self.model}")
                
                # Increase max_tokens for Gemini to avoid truncation
                max_tokens = 16000 if 'gemini' in self.model.lower() else 8000
                
                facts_text = self.client.generate_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.2,
                    max_tokens=max_tokens
                )
                
                # Use the helper method to clean and parse the JSON
                facts = self._clean_llm_output(facts_text)
                
                if self.verbose:
                    print(f"[INFO] Extracted {len(facts)} facts from chunk {i+1}")
                
                all_facts.extend(facts)
            except Exception as e:
                print(f"[ERROR] Error processing chunk {i+1}: {str(e)}")
                if self.verbose:
                    print(f"[DEBUG] First 500 chars of response: {facts_text[:500] if 'facts_text' in locals() else 'N/A'}")
                continue
                
        # Deduplicate facts
        unique_facts = []
        seen_requirements = set()
        
        for fact in all_facts:
            req = fact.get("requirement", "")
            if req and req not in seen_requirements:
                seen_requirements.add(req)
                unique_facts.append(fact)
        
        if self.verbose:
            print(f"[INFO] Total facts extracted: {len(all_facts)}")
            print(f"[INFO] Unique facts after deduplication: {len(unique_facts)}")
        
        # Limit the number of facts if specified
        if max_facts and len(unique_facts) > max_facts:
            if self.verbose:
                print(f"[INFO] Limiting to {max_facts} facts as requested")
            return unique_facts[:max_facts]
                
        return unique_facts

def process_document_with_llm(document, model=DEFAULT_OPENAI_MODEL, use_gemini=False):
    """
    Use LLM to directly split and analyze a document in one go.
    
    Args:
        document (str): The document text to analyze
        model (str): Model to use for analysis
        use_gemini (bool): Whether to use Gemini instead of OpenAI
        
    Returns:
        Dict or None: Structured analysis of the document
    """
    # Create appropriate client
    if use_gemini or 'gemini' in model.lower():
        if not GEMINI_AVAILABLE:
            raise ImportError("To use Gemini, install the package: pip install google-generativeai")
        client = GeminiClient(model=model if 'gemini' in model.lower() else DEFAULT_GEMINI_MODEL)
        # Gemini can handle much more text
        max_doc_length = 200000
    else:
        client = OpenAIClient(model=model)
        max_doc_length = 75000  # For OpenAI models
    
    # Truncate document if needed
    doc_text = document[:max_doc_length]
    
    prompt = f"""
    Analyze this Medicare documentation and extract:
    1. Key sections and their boundaries
    2. Important requirements and rules
    3. Required documentation elements
    
    Return a JSON object with:
    - "sections": array of section objects with "title" and "content"
    - "requirements": array of requirement objects with "topic", "rule", and "documentation_elements"
    
    Document:
    {doc_text}
    """
    
    system_prompt = "You are a Medicare documentation expert who can analyze and structure complex policies."
    
    try:
        analysis_text = client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=8000 if use_gemini else 4000
        )
        
        analysis_text = analysis_text.replace("```json", "").replace("```", "").strip()
        analysis = json.loads(analysis_text)
        return analysis
    except Exception as e:
        print(f"Error in document analysis: {str(e)}")
        return None

def create_evaluation_dataset(document, method="independent", model=DEFAULT_OPENAI_MODEL, num_questions=10, use_gemini=False):
    """
    Create evaluation dataset using specified method.
    
    Parameters:
    - document: Source document text
    - method: "independent" (uses IndependentKnowledgeSource), "direct" (uses process_document_with_llm),
              or "hybrid" (combines both)
    - model: LLM model to use
    - num_questions: Number of questions to generate
    - use_gemini: Whether to use Gemini instead of OpenAI (recommended for large documents)
    
    Returns:
    - List of QA pairs
    """
    # Configure client based on provider selection
    if use_gemini:
        if not GEMINI_AVAILABLE:
            raise ImportError("To use Gemini, install the package: pip install google-generativeai")
        gemini_model = DEFAULT_GEMINI_MODEL if 'gemini' not in model.lower() else model
        client = GeminiClient(model=gemini_model)
    else:
        client = OpenAIClient(model=model)
    
    if method == "independent" or method == "hybrid":
        # Use IndependentKnowledgeSource approach
        knowledge_source = IndependentKnowledgeSource(
            document, 
            model=model, 
            client=client,
            use_gemini=use_gemini
        )
        facts = knowledge_source.extract_key_facts(max_facts=20)  # Limit facts for efficiency
        
        if facts:
            if method == "hybrid":
                # In hybrid mode, supplement with direct analysis
                direct_analysis = process_document_with_llm(document, model, use_gemini=use_gemini)
                if direct_analysis and "requirements" in direct_analysis:
                    # Convert direct analysis requirements to same format as facts
                    for req in direct_analysis.get("requirements", []):
                        fact = {
                            "topic": req.get("topic", "General"),
                            "requirement": req.get("rule", ""),
                            "documentation_elements": req.get("documentation_elements", []),
                            "exceptions": req.get("exceptions", [])
                        }
                        # Check if this is a new requirement
                        if fact["requirement"] and fact["requirement"] not in [f["requirement"] for f in facts]:
                            facts.append(fact)
            
            return generate_questions_from_facts(facts, model, num_questions, client=client)
    
    # Direct method or fallback
    return generate_questions_direct(document, model, num_questions, use_gemini=use_gemini)

def create_gemini_evaluation(document, num_questions=10):
    """
    Convenience function to create evaluation dataset using Gemini's large context window.
    This is particularly useful for large documents that would require excessive chunking with GPT models.
    
    Parameters:
    - document: Source document text (can be very large, Gemini handles up to ~1M tokens)
    - num_questions: Number of questions to generate
    
    Returns:
    - List of QA pairs
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("To use Gemini, install the package: pip install google-generativeai")
        
    return create_evaluation_dataset(
        document,
        method="hybrid",  # Use hybrid approach for best results
        model=DEFAULT_GEMINI_MODEL,
        num_questions=num_questions,
        use_gemini=True
    )

def generate_questions_from_facts(facts, model=DEFAULT_OPENAI_MODEL, num_questions=10, client=None, use_gemini=False):
    """Generate questions based on extracted facts"""
    # Create appropriate client if not provided
    if client is None:
        if use_gemini or 'gemini' in model.lower():
            if not GEMINI_AVAILABLE:
                raise ImportError("To use Gemini, install the package: pip install google-generativeai")
            client = GeminiClient(model=model if 'gemini' in model.lower() else DEFAULT_GEMINI_MODEL)
        else:
            client = OpenAIClient(model=model)
    
    # Create a helper to clean JSON output
    def clean_json_output(text):
        """Clean and parse JSON from LLM outputs"""
        cleaned_text = text.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parse error: {str(e)}, attempting to fix...")
            
            try:
                # Fix 1: Add missing quotes (if odd number of quotes)
                if cleaned_text.count('"') % 2 == 1:
                    cleaned_text += '"'
                
                # Fix 2: Add missing closing brackets
                if cleaned_text.count('[') > cleaned_text.count(']'):
                    cleaned_text += ']' * (cleaned_text.count('[') - cleaned_text.count(']'))
                if cleaned_text.count('{') > cleaned_text.count('}'):
                    cleaned_text += '}' * (cleaned_text.count('{') - cleaned_text.count('}'))
                
                parsed = json.loads(cleaned_text)
                print("[INFO] Successfully fixed JSON")
                return parsed
            except:
                print("[ERROR] Could not repair JSON, returning empty list")
                return []
    
    # Group facts by topic
    topics = {}
    for fact in facts:
        topic = fact.get("topic", "General")
        if topic not in topics:
            topics[topic] = []
        topics[topic].append(fact)
    
    # Generate a balanced set of questions across topics
    questions = []
    system_prompt = "You create challenging Medicare documentation exam questions."
    
    for topic, topic_facts in topics.items():
        # Calculate how many questions to allocate to this topic
        topic_question_count = max(1, int(num_questions * (len(topic_facts) / len(facts))))
        
        prompt = f"""
        Create {topic_question_count} challenging but realistic Medicare documentation questions about this topic:
        
        Topic: {topic}
        
        Facts:
        {json.dumps(topic_facts)}
        
        For each question:
        1. Make it specific and test important Medicare documentation knowledge
        2. Provide an expert-level answer that's complete and accurate
        3. Format as a JSON array with "question" and "example_answer" fields
        """
        
        try:
            # Increase max_tokens for Gemini to avoid truncation
            max_tokens = 16000 if use_gemini or 'gemini' in model.lower() else 8000
            
            qa_text = client.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            topic_questions = clean_json_output(qa_text)
            questions.extend(topic_questions)
        except Exception as e:
            print(f"Error generating questions for topic {topic}: {str(e)}")
    
    # If we don't have enough questions, generate more general ones
    if len(questions) < num_questions:
        additional_needed = num_questions - len(questions)
        general_prompt = f"""
        Create {additional_needed} more challenging Medicare documentation questions based on these facts:
        
        Facts:
        {json.dumps(facts)}
        
        Format as a JSON array with "question" and "example_answer" fields.
        """
        
        try:
            max_tokens = 16000 if use_gemini or 'gemini' in model.lower() else 8000
            
            qa_text = client.generate_completion(
                prompt=general_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=max_tokens
            )
            
            additional_questions = clean_json_output(qa_text)
            questions.extend(additional_questions)
        except Exception as e:
            print(f"Error generating additional questions: {str(e)}")
    
    # Limit to requested number
    return questions[:num_questions]

def clean_json_output(text):
    """Helper function to clean and parse JSON from LLM outputs using stack-based approach"""
    cleaned_text = text.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"[WARNING] JSON parse error: {str(e)}, attempting to fix...")
        
        try:
            # Fix unterminated string issue first (common with LLMs)
            if cleaned_text.count('"') % 2 == 1:
                # Find the last opening quote without a closing quote
                last_idx = cleaned_text.rindex('"')
                if last_idx > 0 and cleaned_text[last_idx-1] != '\\':  # Not an escaped quote
                    cleaned_text += '"'  # Add missing closing quote
            
            # Stack-based approach to fix bracket mismatches
            stack = []
            valid_pairs = {'{': '}', '[': ']', '"': '"'}
            # We only track structural brackets, not quotes inside string literals
            in_string = False
            escaped = False
            fixed_text = cleaned_text
            
            # First pass: check for proper structure and identify mismatches
            for i, char in enumerate(cleaned_text):
                if char == '\\' and not escaped:
                    escaped = True
                    continue
                
                if char == '"' and not escaped:
                    in_string = not in_string
                
                if not in_string:  # Only process brackets outside of strings
                    if char in '{[':
                        stack.append(char)
                    elif char in '}]':
                        if not stack:  # Extra closing bracket
                            # This is an error, but we'll leave it for now
                            pass
                        else:
                            opening = stack.pop()
                            if valid_pairs[opening] != char:  # Mismatched brackets
                                pass  # We'll handle this in the repair phase
                
                escaped = False if escaped else escaped
            
            # Second pass: add missing closing brackets based on stack content
            if stack:
                # Add missing closing brackets in reverse order
                closing_brackets = ''.join(valid_pairs[bracket] for bracket in reversed(stack))
                fixed_text += closing_brackets
                print(f"[INFO] Adding missing closing brackets: {closing_brackets}")
            
            # Try to parse the fixed JSON
            parsed = json.loads(fixed_text)
            print("[INFO] Successfully fixed JSON using stack-based approach")
            return parsed
        except Exception as e:
            print(f"[ERROR] Could not repair JSON: {str(e)}")
            
            # Fallback: try more aggressive approaches
            try:
                # If we're dealing with an array, try to extract complete objects
                if cleaned_text.strip().startswith('['):
                    import re
                    # Match complete JSON objects
                    items = re.findall(r'(\{.*?\})', cleaned_text)
                    if items:
                        return json.loads('[' + ','.join(items) + ']')
                
                print("[ERROR] All JSON repair attempts failed")
                return []
            except:
                return []

def generate_questions_direct(document, model=DEFAULT_OPENAI_MODEL, num_questions=10, use_gemini=False):
    """Generate questions directly from document - fallback method"""
    # Create appropriate client
    if use_gemini or 'gemini' in model.lower():
        if not GEMINI_AVAILABLE:
            raise ImportError("To use Gemini, install the package: pip install google-generativeai")
        client = GeminiClient(model=model if 'gemini' in model.lower() else DEFAULT_GEMINI_MODEL)
        # Gemini can process much more text at once
        chunk_size = 60000
        max_doc_length = 100000
        max_tokens = 16000  # Increase max tokens for Gemini
    else:
        client = OpenAIClient(model=model)
        # OpenAI has smaller context limits
        chunk_size = 8000
        max_doc_length = 50000
        max_tokens = 4000
    
    # If document is too large, take first chunk
    doc_text = document[:max_doc_length]
    
    prompt = f"""
    You are a Medicare documentation expert creating an evaluation dataset.
    
    Based on this documentation, create {num_questions} challenging but realistic questions that test knowledge of:
    1. Medicare documentation requirements
    2. Specific coding rules
    3. Required elements for different services
    
    For each question, provide an expert-level answer that's complete and accurate.
    
    Format as a JSON array with "question" and "example_answer" fields.
    
    Document:
    {doc_text}
    """
    
    system_prompt = "You create challenging Medicare documentation exam questions."
    
    try:
        qa_text = client.generate_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=max_tokens
        )
        
        questions = clean_json_output(qa_text)
        
        return questions[:num_questions]
    except Exception as e:
        print(f"Error generating questions directly: {str(e)}")
        return []


class EnhancedRAPTOREvaluator:
    def __init__(self, qa_pairs, ra_instance, source_document=None, model="gpt-4o", temperature=0.0, use_gemini=True, verbose=False):
        """
        Enhanced evaluator for RAPTOR with multiple evaluation strategies.
        
        Parameters:
        - qa_pairs: List of dictionaries with "question" and "example_answer" keys.
        - ra_instance: RAPTOR instance used for retrievals.
        - source_document: Original document text (optional) for ground truth comparison.
        - model: LLM model for evaluation (default "gpt-4o").
        - temperature: Temperature setting (default 0.0).
        - use_gemini: Whether to use Gemini instead of OpenAI (recommended for large documents).
        - verbose: Whether to print detailed logs.
        """
        self.qa_pairs = qa_pairs
        self.ra = ra_instance
        self.source_document = source_document
        self.model = model
        self.temperature = temperature
        self.verbose = verbose
        
        # Create appropriate client
        if use_gemini or 'gemini' in model.lower():
            if not GEMINI_AVAILABLE:
                raise ImportError("To use Gemini, install the package: pip install google-generativeai")
            self.client = GeminiClient(model=model if 'gemini' in model.lower() else DEFAULT_GEMINI_MODEL, verbose=verbose)
        else:
            self.client = OpenAIClient(model=model)
        
        # Create a reference knowledge base if source document is provided
        self.reference_facts = None
        if source_document:
            if self.verbose:
                print(f"[INFO] Creating reference knowledge base from source document ({len(source_document)} chars)")
            knowledge_source = IndependentKnowledgeSource(
                source_document, 
                model=model,
                client=self.client,
                use_gemini=use_gemini,
                verbose=verbose
            )
            self.reference_facts = knowledge_source.extract_key_facts()
            if self.verbose:
                print(f"[INFO] Extracted {len(self.reference_facts)} reference facts")
        
    def evaluate_with_llm(self):
        """
        Evaluates RAPTOR using LLM-based metrics.
        """
        if self.verbose:
            print(f"[INFO] Starting LLM-based evaluation of {len(self.qa_pairs)} QA pairs")
            
        qa_results = []
        for i, qa in enumerate(self.qa_pairs):
            question = qa["question"]
            example_answer = qa["example_answer"]
            
            if self.verbose:
                print(f"[INFO] Processing QA pair {i+1}/{len(self.qa_pairs)}")
                print(f"[INFO] Question: {question[:100]}...")
            
            # Get RAPTOR's answer
            retrieved_answer = self.ra.answer_question(question=question)
            
            if self.verbose:
                print(f"[INFO] Retrieved answer: {len(retrieved_answer)} chars")
            
            # Enhanced evaluation prompt with more specific Medicare focus
            prompt = f"""
            Evaluate this Medicare documentation Q&A pair against the example answer.
            
            Scoring Guidelines (0-10):
            - Accuracy: Correctness according to Medicare guidelines and policies
            - Completeness: Inclusion of all required documentation elements
            - Faithfulness: Adherence to source document information without fabrication
            - Relevance: Focus on the specific Medicare documentation requirements
            
            Return only a JSON object:
            {{"feedback": "brief_actionable_feedback", 
              "metrics": {{"accuracy": int, "completeness": int, "faithfulness": int, "relevance": int}},
              "missing_elements": ["list_of_missing_key_elements"],
              "incorrect_elements": ["list_of_incorrect_elements"]
            }}
            
            Question: {question}
            Example Good Answer: {example_answer}
            Retrieved Answer: {retrieved_answer}
            """

            try:
                if self.verbose:
                    print(f"[INFO] Requesting evaluation from {self.client.get_provider()} model")
                
                evaluation_text = self.client.generate_completion(
                    prompt=prompt,
                    system_prompt="You are an expert Medicare documentation evaluator. Return only JSON.",
                    temperature=self.temperature
                )
                
                # Handle potential JSON formatting issues
                evaluation_text = evaluation_text.replace("```json", "").replace("```", "").strip()
                evaluation_data = json.loads(evaluation_text)
                
                if self.verbose:
                    print(f"[INFO] Evaluation complete - Accuracy: {evaluation_data['metrics']['accuracy']}, Completeness: {evaluation_data['metrics']['completeness']}")
                
                qa_result = {
                    "question": question,
                    "example_answer": example_answer,
                    "retrieved_answer": retrieved_answer,
                    "accuracy": evaluation_data["metrics"]["accuracy"],
                    "completeness": evaluation_data["metrics"]["completeness"],
                    "faithfulness": evaluation_data["metrics"]["faithfulness"],
                    "relevance": evaluation_data["metrics"]["relevance"],
                    "feedback": evaluation_data["feedback"],
                    "missing_elements": evaluation_data.get("missing_elements", []),
                    "incorrect_elements": evaluation_data.get("incorrect_elements", [])
                }
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Evaluation failed: {str(e)}")
                
                qa_result = {
                    "question": question,
                    "example_answer": example_answer,
                    "retrieved_answer": retrieved_answer,
                    "accuracy": None,
                    "completeness": None,
                    "faithfulness": None,
                    "relevance": None,
                    "feedback": f"Error parsing evaluation: {str(e)}",
                    "missing_elements": [],
                    "incorrect_elements": []
                }
            qa_results.append(qa_result)

        return pd.DataFrame(qa_results)
    
    def evaluate_with_keywords(self):
        """
        Evaluates based on keyword presence in both example and retrieved answers.
        """
        if self.verbose:
            print(f"[INFO] Starting keyword-based evaluation of {len(self.qa_pairs)} QA pairs")
            
        qa_results = []
        for i, qa in enumerate(self.qa_pairs):
            question = qa["question"]
            example_answer = qa["example_answer"]
            
            if self.verbose:
                print(f"[INFO] Processing QA pair {i+1}/{len(self.qa_pairs)} for keyword extraction")
            
            # Get RAPTOR's answer
            retrieved_answer = self.ra.answer_question(question=question)
            
            # Extract keywords from example answer
            prompt = f"""
            Extract 5-10 critical Medicare documentation keywords/phrases from this answer:
            
            {example_answer}
            
            Return ONLY a JSON array of strings: ["keyword1", "keyword2", ...]
            """
            
            try:
                if self.verbose:
                    print(f"[INFO] Extracting keywords from example answer")
                
                keywords_text = self.client.generate_completion(
                    prompt=prompt,
                    system_prompt="Extract Medicare documentation keywords. Return only JSON.",
                    temperature=0.2
                )
                
                keywords_text = keywords_text.replace("```json", "").replace("```", "").strip()
                keywords = json.loads(keywords_text)
                
                if self.verbose:
                    print(f"[INFO] Extracted {len(keywords)} keywords")
                
                # Calculate keyword match rate
                matches = sum(1 for kw in keywords if kw.lower() in retrieved_answer.lower())
                match_rate = matches / len(keywords) if keywords else 0
                
                if self.verbose:
                    print(f"[INFO] Keyword match rate: {match_rate:.2f} ({matches}/{len(keywords)})")
                
                qa_result = {
                    "question": question,
                    "example_answer": example_answer,
                    "retrieved_answer": retrieved_answer,
                    "keyword_match_rate": match_rate,
                    "keywords": keywords,
                    "matched_keywords": [kw for kw in keywords if kw.lower() in retrieved_answer.lower()]
                }
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Keyword extraction failed: {str(e)}")
                    
                qa_result = {
                    "question": question,
                    "example_answer": example_answer, 
                    "retrieved_answer": retrieved_answer,
                    "keyword_match_rate": None,
                    "keywords": [],
                    "matched_keywords": [],
                    "error": str(e)
                }
            qa_results.append(qa_result)
            
        return pd.DataFrame(qa_results)
    
    def evaluate_fact_alignment(self):
        """
        Evaluates alignment with extracted facts from source document.
        Only works when source_document was provided during initialization.
        """
        if not self.reference_facts:
            if self.verbose:
                print(f"[WARNING] No reference facts available - skipping fact alignment evaluation")
            return pd.DataFrame()
            
        if self.verbose:
            print(f"[INFO] Starting fact alignment evaluation with {len(self.reference_facts)} reference facts")
            
        qa_results = []
        for i, qa in enumerate(self.qa_pairs):
            question = qa["question"]
            
            if self.verbose:
                print(f"[INFO] Processing QA pair {i+1}/{len(self.qa_pairs)} for fact alignment")
            
            retrieved_answer = self.ra.answer_question(question=question)
            
            # Find relevant facts from reference knowledge
            prompt = f"""
            Identify which of these Medicare documentation facts are relevant to the question:
            
            Question: {question}
            
            Facts:
            {json.dumps(self.reference_facts)}
            
            Return only the indices of relevant facts as a JSON array of integers.
            """
            
            try:
                if self.verbose:
                    print(f"[INFO] Identifying relevant facts for question")
                
                indices_text = self.client.generate_completion(
                    prompt=prompt,
                    system_prompt="You identify relevant Medicare facts for evaluation.",
                    temperature=0.2
                )
                
                indices_text = indices_text.replace("```json", "").replace("```", "").strip()
                relevant_indices = json.loads(indices_text)
                
                # Get relevant facts
                relevant_facts = [self.reference_facts[i] for i in relevant_indices if i < len(self.reference_facts)]
                
                if self.verbose:
                    print(f"[INFO] Found {len(relevant_facts)} relevant facts")
                
                if not relevant_facts:
                    if self.verbose:
                        print(f"[WARNING] No relevant facts found for this question")
                        
                    qa_result = {
                        "question": question,
                        "retrieved_answer": retrieved_answer,
                        "fact_coverage": 0.0,
                        "fact_accuracy": 0.0
                    }
                    qa_results.append(qa_result)
                    continue
                
                # Evaluate against relevant facts
                eval_prompt = f"""
                Evaluate how well this answer covers the relevant Medicare documentation facts:
                
                Question: {question}
                Answer: {retrieved_answer}
                
                Relevant Facts:
                {json.dumps(relevant_facts)}
                
                Return a JSON object with:
                - "fact_coverage": percentage (0-1) of relevant facts covered
                - "fact_accuracy": accuracy score (0-1) of covered facts
                - "missing_facts": array of facts not adequately covered
                - "inaccuracies": array of inaccurate statements relative to facts
                """
                
                if self.verbose:
                    print(f"[INFO] Evaluating answer against relevant facts")
                
                eval_text = self.client.generate_completion(
                    prompt=eval_prompt,
                    system_prompt="You evaluate Medicare documentation answers against facts.",
                    temperature=0.2
                )
                
                eval_text = eval_text.replace("```json", "").replace("```", "").strip()
                eval_data = json.loads(eval_text)
                
                if self.verbose:
                    print(f"[INFO] Fact coverage: {eval_data['fact_coverage']:.2f}, Fact accuracy: {eval_data['fact_accuracy']:.2f}")
                
                qa_result = {
                    "question": question,
                    "retrieved_answer": retrieved_answer,
                    "fact_coverage": eval_data["fact_coverage"],
                    "fact_accuracy": eval_data["fact_accuracy"],
                    "missing_facts": eval_data.get("missing_facts", []),
                    "inaccuracies": eval_data.get("inaccuracies", []),
                    "relevant_facts": relevant_facts
                }
                
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Fact alignment evaluation failed: {str(e)}")
                    
                qa_result = {
                    "question": question,
                    "retrieved_answer": retrieved_answer,
                    "fact_coverage": None,
                    "fact_accuracy": None,
                    "error": str(e)
                }
                
            qa_results.append(qa_result)
            
        return pd.DataFrame(qa_results)
        
    def expert_blind_evaluation(self):
        """
        Uses a different LLM to independently answer questions and 
        evaluate against RAPTOR without seeing expected answers.
        """
        if self.verbose:
            print(f"[INFO] Starting expert blind evaluation of {len(self.qa_pairs)} QA pairs")
            
        qa_results = []
        
        for i, qa in enumerate(self.qa_pairs):
            question = qa["question"]
            example_answer = qa["example_answer"]
            
            if self.verbose:
                print(f"[INFO] Processing QA pair {i+1}/{len(self.qa_pairs)} for expert blind evaluation")
            
            retrieved_answer = self.ra.answer_question(question=question)
            
            # Get independent expert answer
            expert_prompt = f"""
            As a Medicare documentation expert, answer this question:
            
            {question}
            
            Provide a detailed, accurate answer according to Medicare guidelines.
            """
            
            try:
                # Use a different model if available, otherwise use same model
                expert_model = "gpt-4" if self.model != "gpt-4" else "gpt-4o"
                if 'gemini' in self.model.lower():
                    expert_model = "gemini-2.0-flash" if self.model != "gemini-2.0-flash" else DEFAULT_GEMINI_MODEL
                
                if self.verbose:
                    print(f"[INFO] Generating independent expert answer using {expert_model}")
                
                # Create a separate client for the expert model
                if 'gemini' in expert_model.lower():
                    expert_client = GeminiClient(model=expert_model, verbose=self.verbose)
                else:
                    expert_client = OpenAIClient(model=expert_model)
                
                expert_answer = expert_client.generate_completion(
                    prompt=expert_prompt,
                    system_prompt="You are a Medicare documentation expert.",
                    temperature=0.1
                )
                
                if self.verbose:
                    print(f"[INFO] Generated expert answer: {len(expert_answer)} chars")
                    print(f"[INFO] Comparing retrieved answer with expert answer")
                
                # Compare retrieved answer with independent expert answer
                compare_prompt = f"""
                Compare these two Medicare documentation answers for the question:
                
                Question: {question}
                
                Answer A: {retrieved_answer}
                
                Answer B: {expert_answer}
                
                Return a JSON object with:
                - "agreement_score": 0-1 value indicating factual agreement
                - "coverage_comparison": 0-1 value indicating relative coverage
                - "critical_discrepancies": array of critical factual disagreements
                - "unique_to_a": array of relevant points only in Answer A
                - "unique_to_b": array of relevant points only in Answer B
                - "overall_assessment": brief assessment of which is more accurate/complete
                """
                
                eval_text = self.client.generate_completion(
                    prompt=compare_prompt,
                    system_prompt="You are evaluating Medicare documentation answers.",
                    temperature=0.2
                )
                
                eval_text = eval_text.replace("```json", "").replace("```", "").strip()
                eval_data = json.loads(eval_text)
                
                if self.verbose:
                    print(f"[INFO] Agreement score: {eval_data['agreement_score']:.2f}, Coverage comparison: {eval_data['coverage_comparison']:.2f}")
                
                qa_result = {
                    "question": question,
                    "retrieved_answer": retrieved_answer,
                    "expert_answer": expert_answer,
                    "agreement_score": eval_data["agreement_score"],
                    "coverage_comparison": eval_data["coverage_comparison"],
                    "critical_discrepancies": eval_data.get("critical_discrepancies", []),
                    "unique_to_raptor": eval_data.get("unique_to_a", []),
                    "unique_to_expert": eval_data.get("unique_to_b", []),
                    "overall_assessment": eval_data.get("overall_assessment", "")
                }
                
            except Exception as e:
                if self.verbose:
                    print(f"[ERROR] Expert blind evaluation failed: {str(e)}")
                    
                qa_result = {
                    "question": question,
                    "retrieved_answer": retrieved_answer,
                    "error": str(e)
                }
                
            qa_results.append(qa_result)
            
        return pd.DataFrame(qa_results)
    
    def perform_comprehensive_evaluation(self):
        """
        Runs multiple evaluation methods and combines results.
        """
        if self.verbose:
            print(f"[INFO] Starting comprehensive evaluation")
            
        llm_results = self.evaluate_with_llm()
        
        if self.verbose:
            print(f"[INFO] LLM evaluation complete, running keyword evaluation")
            
        keyword_results = self.evaluate_with_keywords()
        
        # Initialize combined results
        combined_results = llm_results.copy()
        combined_results['keyword_match_rate'] = keyword_results['keyword_match_rate']
        combined_results['keywords'] = keyword_results['keywords']
        combined_results['matched_keywords'] = keyword_results['matched_keywords']
        
        # Add fact alignment if source document was provided
        if self.source_document:
            if self.verbose:
                print(f"[INFO] Source document provided, running fact alignment evaluation")
                
            fact_results = self.evaluate_fact_alignment()
            if not fact_results.empty:
                combined_results['fact_coverage'] = fact_results['fact_coverage'].values
                combined_results['fact_accuracy'] = fact_results['fact_accuracy'].values
        else:
            # If no source document, use expert blind evaluation
            if self.verbose:
                print(f"[INFO] No source document, running expert blind evaluation")
                
            expert_results = self.expert_blind_evaluation()
            if not expert_results.empty:
                combined_results['agreement_score'] = expert_results['agreement_score'].values
                combined_results['coverage_comparison'] = expert_results['coverage_comparison'].values
        
        # Calculate aggregate score (weighted) based on available metrics
        if self.verbose:
            print(f"[INFO] Calculating aggregate scores")
            
        if self.source_document and 'fact_coverage' in combined_results.columns:
            combined_results['aggregate_score'] = (
                combined_results['accuracy'] * 0.25 + 
                combined_results['completeness'] * 0.20 + 
                combined_results['faithfulness'] * 0.15 + 
                combined_results['relevance'] * 0.10 + 
                combined_results['keyword_match_rate'] * 10 * 0.10 +  # Scale to 0-10
                combined_results['fact_coverage'] * 10 * 0.10 +  # Scale to 0-10
                combined_results['fact_accuracy'] * 10 * 0.10   # Scale to 0-10
            )
        elif 'agreement_score' in combined_results.columns:
            combined_results['aggregate_score'] = (
                combined_results['accuracy'] * 0.25 + 
                combined_results['completeness'] * 0.20 + 
                combined_results['faithfulness'] * 0.15 + 
                combined_results['relevance'] * 0.10 + 
                combined_results['keyword_match_rate'] * 10 * 0.15 +  # Scale to 0-10
                combined_results['agreement_score'] * 10 * 0.15   # Scale to 0-10
            )
        else:
            combined_results['aggregate_score'] = (
                combined_results['accuracy'] * 0.3 + 
                combined_results['completeness'] * 0.3 + 
                combined_results['faithfulness'] * 0.2 + 
                combined_results['relevance'] * 0.1 + 
                combined_results['keyword_match_rate'] * 10 * 0.1  # Scale to 0-10
            )
        
        if self.verbose:
            print(f"[INFO] Comprehensive evaluation complete")
            avg_score = combined_results['aggregate_score'].mean()
            print(f"[INFO] Average aggregate score: {avg_score:.2f}/10")
            
        return combined_results
    
    def generate_evaluation_report(self):
        """
        Evaluates and generates a comprehensive report with visualizations.
        """
        if self.verbose:
            print(f"[INFO] Generating evaluation report")
            
        results = self.perform_comprehensive_evaluation()
        
        # Convert aggregate_score to numeric, coercing errors to NaN
        results['aggregate_score'] = pd.to_numeric(results['aggregate_score'], errors='coerce')
        
        # Calculate summary statistics (safely handling NaN values)
        summary = {
            'avg_accuracy': results['accuracy'].mean() if 'accuracy' in results else None,
            'avg_completeness': results['completeness'].mean() if 'completeness' in results else None,
            'avg_faithfulness': results['faithfulness'].mean() if 'faithfulness' in results else None,
            'avg_relevance': results['relevance'].mean() if 'relevance' in results else None,
            'avg_keyword_match': results['keyword_match_rate'].mean() if 'keyword_match_rate' in results else None,
            'avg_aggregate_score': results['aggregate_score'].mean() if not results['aggregate_score'].isna().all() else None,
            'question_count': len(results),
            'perfect_score_count': sum(results['aggregate_score'] >= 9.0) if not results['aggregate_score'].isna().all() else 0,
            'poor_score_count': sum(results['aggregate_score'] < 6.0) if not results['aggregate_score'].isna().all() else 0
        }
        
        # Identify common missing elements (safely)
        all_missing = []
        if 'missing_elements' in results:
            for missing_list in results['missing_elements']:
                if isinstance(missing_list, list):
                    all_missing.extend(missing_list)
        
        missing_counts = pd.Series(all_missing).value_counts() if all_missing else pd.Series([])
        
        # Create a report (safely handling potential errors)
        report = {
            'summary_stats': summary,
            'common_missing_elements': missing_counts.to_dict(),
            'detailed_results': results.to_dict('records')  # Convert entire results to dict format
        }
        
        # Only add best/worst questions if we have valid aggregate scores
        if not results['aggregate_score'].isna().all():
            try:
                report['worst_performing_questions'] = results.nsmallest(
                    min(3, len(results)), 'aggregate_score'
                )[['question', 'aggregate_score', 'feedback']].to_dict('records')
                
                report['best_performing_questions'] = results.nlargest(
                    min(3, len(results)), 'aggregate_score'
                )[['question', 'aggregate_score', 'feedback']].to_dict('records')
            except Exception as e:
                if self.verbose:
                    print(f"[WARNING] Could not determine best/worst questions: {str(e)}")
                # Fallback: just include all questions
                report['all_questions'] = results[['question', 'aggregate_score', 'feedback']].to_dict('records')
        
        if self.verbose:
            print(f"[INFO] Report generated with {len(results)} evaluated questions")
            if 'avg_aggregate_score' in summary and summary['avg_aggregate_score'] is not None:
                print(f"[INFO] Average aggregate score: {summary['avg_aggregate_score']:.2f}/10")
            if 'common_missing_elements' in report:
                print(f"[INFO] Found {len(report['common_missing_elements'])} common missing elements")
        
        return report
        
    def plot_evaluation_results(self, results=None):
        """
        Creates visualizations of the evaluation results.
        """
        if results is None:
            if self.verbose:
                print(f"[INFO] No results provided, running comprehensive evaluation")
                
            results = self.perform_comprehensive_evaluation()
            
        if self.verbose:
            print(f"[INFO] Creating evaluation visualizations")
            
        # Prepare metrics for visualization
        metrics = ['accuracy', 'completeness', 'faithfulness', 'relevance', 'keyword_match_rate']
        avg_metrics = [results[m].mean() for m in metrics]
        
        # Convert keyword_match_rate to 0-10 scale
        avg_metrics[-1] = avg_metrics[-1] * 10
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Metrics histogram
        results_for_plot = results[metrics].copy()
        results_for_plot['keyword_match_rate'] = results_for_plot['keyword_match_rate'] * 10
        results_for_plot.columns = ['Accuracy', 'Completeness', 'Faithfulness', 'Relevance', 'Keyword Match']
        results_for_plot.plot(kind='bar', ax=ax1)
        ax1.set_title('Evaluation Metrics by Question')
        ax1.set_ylim(0, 10)
        ax1.set_ylabel('Score (0-10)')
        ax1.set_xlabel('Question Index')
        
        # Scatter plot of accuracy vs. completeness
        ax2.scatter(results['accuracy'], results['completeness'], s=100, alpha=0.7)
        for i, (_, row) in enumerate(results.iterrows()):
            ax2.annotate(str(i), (row['accuracy'], row['completeness']))
        ax2.set_title('Accuracy vs. Completeness')
        ax2.set_xlabel('Accuracy Score')
        ax2.set_ylabel('Completeness Score')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if self.verbose:
            print(f"[INFO] Visualization complete")
            
        return fig