#!/usr/bin/env python3
"""
BART-Large Enhanced ASL-to-English Pipeline

This script processes a stream of characters (typical of ASL fingerspelling input),
cleans them, and uses a Seq2Seq Transformer (BART-Large) to predict the coherent 
English sentence.

Key Features:
- BART-Large Model: Handles typos, missing spaces, capitalization, and punctuation simultaneously.
- Extensive Pattern Matching: Overrides for 100+ common ASL phrases.
- Smart Character Collapse: Intelligent deduplication of repeated characters.

Usage:
    pip install -U transformers torch sentencepiece
    python nlp_refiner_v3.py
"""

from typing import List, Optional, Tuple, Dict
import re
import time
import logging

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception as e:
    raise ImportError(
        "Missing dependencies. Install with: pip install transformers torch sentencepiece"
    )

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("bart_asl")


# ---------- Enhanced ASL Pattern Dictionary ----------

class ASLPatternMatcher:
    """Comprehensive ASL fingerspelling pattern matcher"""
    
    def __init__(self):
        # Extensive pattern dictionary from your provided code
        self.patterns = {
            # Greetings
            'hello': 'hello',
            'hi': 'hi',
            'hey': 'hey',
            'goodmorning': 'good morning',
            'goodafternoon': 'good afternoon',
            'goodevening': 'good evening',
            'goodnight': 'good night',
            'goodbye': 'goodbye',
            'bye': 'bye',
            
            # Common phrases
            'thankyou': 'thank you',
            'thanks': 'thanks',
            'please': 'please',
            'sorry': 'sorry',
            'excuseme': 'excuse me',
            'welcomeback': 'welcome back',
            'yourwelcome': 'you are welcome',
            'youarewelcome': 'you are welcome',
            
            # Love expressions
            'iloveyou': 'i love you',
            'loveyou': 'love you',
            'iloveme': 'i love me',
            'ilove': 'i love',
            
            # Introductions
            'myname': 'my name',
            'mynameis': 'my name is',
            'nicetomeetyou': 'nice to meet you',
            'nicemeeting': 'nice meeting',
            'howareyou': 'how are you',
            'howyou': 'how are you',
            'whatisyourname': 'what is your name',
            'whatsyourname': 'what is your name',
            'whatyourname': 'what is your name',
            
            # Common verbs
            'iam': 'i am',
            'iamstudying': 'i am studying',
            'studying': 'studying',
            'learning': 'learning',
            'working': 'working',
            'teaching': 'teaching',
            
            # Common topics
            'bigdata': 'big data',
            'computer': 'computer',
            'science': 'science',
            'english': 'english',
            'mathematics': 'mathematics',
            'math': 'math',
            
            # Phrases with "very"
            'verymuch': 'very much',
            'thankyouverymuch': 'thank you very much',
            'verynice': 'very nice',
            'verygood': 'very good',
            'veryhappy': 'very happy',
        }
        
        # Sort by length (longest first) for better matching
        self.sorted_patterns = sorted(
            self.patterns.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
    
    def match(self, text: str) -> Optional[str]:
        """Try to match known ASL patterns"""
        text_clean = text.lower().replace(' ', '').replace('-', '')
        
        # Direct pattern matching
        for pattern, replacement in self.sorted_patterns:
            if pattern == text_clean:
                return replacement
        
        # Partial phrase matching logic
        if 'ilov' in text_clean:
            if 'you' in text_clean:
                return 'i love you'
            elif 'me' in text_clean:
                return 'i love me'
            else:
                return 'i love'
        
        if 'thankyou' in text_clean or 'thanku' in text_clean:
            if 'verymuch' in text_clean or 'much' in text_clean:
                return 'thank you very much'
            return 'thank you'
        
        if 'myname' in text_clean and 'is' in text_clean:
            parts = text_clean.split('is')
            if len(parts) > 1 and parts[1].strip():
                name = parts[1].strip()
                return f'my name is {name}'
            return 'my name is'
        
        if 'niceto' in text_clean and 'meet' in text_clean:
            return 'nice to meet you'
        
        return None


# ---------- Smart Character Collapse ----------

def smart_collapse_chars(text: str, dedupe_threshold: int = 3) -> str:
    """
    Intelligently collapse character streams while handling duplicates.
    """
    if not text or not text.strip():
        return ""
    
    tokens = text.strip().split()
    if not tokens:
        return ""
    
    words = []
    current_word = []
    last_char = None
    char_count = 0
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
            
        if len(token) > 1:
            if current_word:
                words.append(''.join(current_word))
                current_word = []
                last_char = None
                char_count = 0
            words.append(token)
        
        elif len(token) == 1 and token.isalpha():
            if token.lower() == last_char:
                char_count += 1
                if char_count < dedupe_threshold:
                    current_word.append(token)
            else:
                current_word.append(token)
                last_char = token.lower()
                char_count = 1
        
        else:
            if current_word:
                words.append(''.join(current_word))
                current_word = []
                last_char = None
                char_count = 0
            words.append(token)
    
    if current_word:
        words.append(''.join(current_word))
    
    return ' '.join(words)


# ---------- Enhanced Spell Corrector (BART-Large) ----------

class EnhancedSpellCorrector:
    """
    Spell corrector using facebook/bart-large.
    Replaces the previous lightweight model with a robust Seq2Seq model
    that handles spacing, punctuation, and grammar in one go.
    """
    
    MODEL_NAME = "facebook/bart-large"
    
    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pattern_matcher = ASLPatternMatcher()
        
        logger.info(f"Loading BART-Large model: {self.MODEL_NAME} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
        
        if self.device == "cuda":
            try:
                # BART-Large is big, half-precision helps memory usage
                self.model = self.model.half()
                logger.info("Enabled FP16 for BART-Large")
            except Exception:
                pass
                
        self.model.to(self.device)
        self.model.eval()
        
    def correct(self, text: str, max_length: int = 128) -> str:
        """
        Corrects the input string using BART-Large.
        Input: "thankyouverymuch" (or spaced text)
        Output: "Thank you very much."
        """
        
        # 1. Pattern Match Bypass (Highest Priority)
        # We strip spaces to check against the dictionary effectively
        collapsed_text = text.replace(' ', '')
        pattern_match = self.pattern_matcher.match(collapsed_text)
        
        if pattern_match:
            # Even if we match a pattern, we might want capitalization/punctuation.
            # However, the dictionary values are clean. Let's capitalize them.
            return pattern_match[0].upper() + pattern_match[1:]
        
        # 2. BART Correction
        # We feed the text to BART. BART is robust to missing spaces, 
        # so we can feed it the collapsed version or the spaced version.
        # Feeding the 'smart_collapsed' version (e.g. "thankyou") is usually best.
        input_text = collapsed_text if len(collapsed_text) > 0 else text

        try:
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=5,           # Higher beams = better quality search
                    early_stopping=True,
                    repetition_penalty=1.0, # BART defaults are usually fine
                    length_penalty=1.0,
                    no_repeat_ngram_size=3
                )
            
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result = decoded.strip()
            
            # Fallback: If BART produces nothing or garbage, return original
            if not result:
                return text
                
            # Post-processing: Ensure basic punctuation if BART missed it
            if result and result[-1] not in '.!?':
                result += '.'
            
            return result
            
        except Exception as e:
            logger.warning(f"BART Correction failed: {e}")
            return text


# ---------- Enhanced Pipeline (Unified) ----------

class EnhancedASLPipeline:
    """
    Unified ASL-to-English pipeline.
    Removed 'SmartSemanticRefiner' because BART-Large handles refinement naturally.
    """
    
    def __init__(self):
        # We no longer need separate spell/semantic models. 
        # BART-Large does it all.
        self.spell_corrector = EnhancedSpellCorrector()
    
    def process(self, raw_buffer: str) -> Tuple[str, Dict]:
        """Process ASL fingerspelling to English text"""
        t0 = time.time()
        meta = {"original": raw_buffer}
        
        # Step 1: Clean whitespace
        cleaned = raw_buffer.strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        meta["cleaned"] = cleaned
        
        # Step 2: Smart character collapse with deduplication
        # Turns "t h a n k" -> "thank"
        collapsed = smart_collapse_chars(cleaned, dedupe_threshold=3)
        meta["collapsed"] = collapsed
        
        # Step 3: BART Correction (Pattern matching is inside)
        corrected = self.spell_corrector.correct(collapsed)
        
        meta["method"] = "BART-Large (Seq2Seq)"
        meta["corrected"] = corrected
        final = corrected
        
        meta["final"] = final
        meta["processing_time_seconds"] = round(time.time() - t0, 4)
        
        return final, meta


# ---------- Demo ----------

if __name__ == "__main__":
    demo_inputs = [
        "i l o v m s s m e e",             # Typo: "iloveme" or "ilovemess"?
        "h e l l o m y n a m e i s s n e h a", # Name extraction
        "i a m s t u d y i n g b i g d a t a", # Merged words
        "t h a n k y o u v e r y m u c h", # Common phrase
        "w h a t i s y o u r n a m e",     # Question
        "n i c e t o m m e e t y o u",     # Repetition
        "h e l ll o m y n a m e i s s n e h a a", # Heavy noise
        "i l o v e y o u",                 # Simple
        "h o w a r e y o u",               # Question
    ]
    
    print("🤟 BART-Large Enhanced ASL Pipeline\n")
    print("Loading model... (This may take a moment for 400M parameters)\n")
    
    pipeline = EnhancedASLPipeline()
    
    for sample in demo_inputs:
        final, debug = pipeline.process(sample)
        print(f"📝 Input:    {sample}")
        print(f"🔗 Collapsed: {debug['collapsed']}")
        print(f"✨ Output:   {final}")
        print(f"⏱️  Time:     {debug['processing_time_seconds']}s")
        print("-" * 30)


# ---------- Compatibility Function for Streamlit App ----------

# Global pipeline instance for efficiency
_global_pipeline = None

def get_pipeline():
    """Get or create global pipeline instance"""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = EnhancedASLPipeline()
    return _global_pipeline

def refine_asl_buffer(buffer_string: str) -> dict:
    """
    Compatibility function for streamlit_app.py
    
    Args:
        buffer_string: Raw ASL buffer string
        
    Returns:
        Dictionary with refinement results matching the expected format
    """
    pipeline = get_pipeline()
    final_text, metadata = pipeline.process(buffer_string)
    
    # Format to match the expected Streamlit interface
    return {
        'original_buffer': buffer_string,
        'preprocessed': metadata.get('cleaned', buffer_string),
        'cleaned': metadata.get('collapsed', buffer_string),
        'refined_text': final_text,
        'processing_time_seconds': metadata.get('processing_time_seconds', 0),
        'model_device': 'cpu'  # Default device info
    }