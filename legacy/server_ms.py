# Server.py File
import grpc
from concurrent import futures
import time
import torch
from TTS.api import TTS
import ollama
import io
import os
import uuid
import re
from functools import lru_cache
from pydub import AudioSegment
from transformers import pipeline, MarianMTModel, MarianTokenizer
import transformers
from proto import story_service_pb2
from proto import story_service_pb2_grpc
import threading

tts_lock = threading.Lock()

transformers.logging.set_verbosity_error()

# ✅ Add this here
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Load TTS model
# tts = TTS(
#     model_path="xtts_model",
#     config_path="xtts_model/config.json",
#     progress_bar=False
# )

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)




tts.to("cuda")

# Load emotion classifier
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

chat_history = []

# Prompts
SHORT_DIALOGUE_PROMPT = """You are a creative storyteller writing for an audio story narration.
Based on the short storyline given, create a simple and emotionally engaging story with a clear beginning, middle, and end, ensuring a natural flow. 
Use concise, easy-to-speak language and express emotions like happiness, sadness, fear, or surprise. 
The story should feel like events are unfolding right now.
IMPORTANT:
- Output only the story. No titles, no comments.
- ONE character dialogue must be included, and the dialogue must be from a female character.
- Story narration must be in THIRD PERSON.
- Dialogue must be in FIRST PERSON.
- The story must be fully concluded — no abrupt endings.
- Story length: ~350 words.
Storyline:"""

MEDIUM_DIALOGUE_PROMPT = """You are a creative storyteller writing for an audio story narration.
Based on the short storyline given, create a simple and emotionally engaging story with a clear beginning, middle, and end, ensuring a natural flow. 
Use concise, easy-to-speak language and express emotions like happiness, sadness, fear, or surprise. 
The story should feel like events are unfolding right now.
IMPORTANT:
- Output only the story. No titles, no comments.
- ONE character dialogue must be included, and the dialogue must be from a female character.
- Story narration must be in THIRD PERSON.
- Dialogue must be in FIRST PERSON.
- The story must be fully concluded — no abrupt endings.
- Story length: 500-700 words.
Storyline:"""

LONG_DIALOGUE_PROMPT = """You are a creative storyteller writing for an audio story narration.
Based on the short storyline given, create a simple and emotionally engaging story with a clear beginning, middle, and end, ensuring a natural flow. 
Use concise, easy-to-speak language and express emotions like happiness, sadness, fear, or surprise. 
The story should feel like events are unfolding right now.
IMPORTANT:
- Output only the story. No titles, no comments.
- ONE character dialogue must be included, and the dialogue must be from a female character.
- Story narration must be in THIRD PERSON.
- Dialogue must be in FIRST PERSON.
- The story must be fully concluded — no abrupt endings.
- Story length: 800-1000 words.
Storyline:"""

SHORT_NARRATION_PROMPT = """You are a creative storyteller writing story narration.
Based on the given storyline, write a very short and emotionally engaging story. The story should be brief, clear, and suitable for single-voice narration. 
Use simple, easy-to-speak language and naturally express emotions like happiness, sadness, fear, surprise, or relief. 
The sentences should be clear, natural, and sound good when read aloud by a TTS system. 
The story should be in third-person.

IMPORTANT:
- Output only the story — no narration or titles.
- Avoid any sort of dialogue inclusion.
- Use expressive, simple language that sounds natural aloud.
- Ensure the story doesn't end abruptly and provides a clean, meaningful conclusion.
- Story Naration in Third Person
- Story length: ~350 words.
- PLEASE MAKE SURE THE STORY HAS A PROPER END

Storyline: """

MEDIUM_NARRATION_PROMPT = """You are a creative storyteller writing story narration.
Based on the given storyline, write a short and emotionally engaging story. 
The story should be brief, clear, and suitable for single-voice narration. 
Use simple, easy-to-speak language to naturally express emotions like happiness, sadness, fear, surprise, or relief. 
Ensure the sentences are clear and natural, sounding good when read aloud by a TTS system. 
The story should be in third person.

IMPORTANT:
- Output only the story — no narration or titles.
- Avoid any sort of dialogue inclusion.
- Use expressive, simple language that sounds natural aloud.
- Ensure the story doesn't end abruptly and provides a clean, meaningful conclusion.
- Story Naration in Third Person
- Story length: ~550–700 words.
- PLEASE MAKE SURE THE STORY HAS A PROPER END

Storyline:
 """

LONG_NARRATION_PROMPT = """You are a creative storyteller writing story narration.
Based on the given storyline, write a short and emotionally engaging story. 
The story should be brief, clear, and suitable for single-voice narration. 
Use simple, easy-to-speak language to naturally express emotions like happiness, sadness, fear, surprise, or relief. 
Ensure the sentences are clear and natural, making them sound good when read aloud by a TTS system. 
The story should be in third person.

IMPORTANT:
- Output only the story — no narration or titles.
- Avoid any sort of dialogue inclusion.
- Use expressive, simple language that sounds natural aloud.
- Ensure the story doesn't end abruptly and provides a clean, meaningful conclusion.
- Story Naration in Third Person
-Story length: ~800–1200+ words.
- PLEASE MAKE SURE THE STORY HAS A PROPER END

Storyline:"""

def sanitize_filename(prompt_text, speaker_display_name, max_length=70):
    prompt_text = re.sub(r'\[PARA_LEVEL:.*?\]', '', prompt_text).strip()
    base = prompt_text.strip().title()
    base = re.sub(r'[^\w\s-]', '', base)
    base = re.sub(r'\s+', ' ', base)
    base = base[:max_length]
    filename = f"{base} - {speaker_display_name}.wav"
    return os.path.join(OUTPUT_DIR, filename)

def detect_leading_silence(sound, silence_thresh=-40, chunk_size=10):
    trim_ms = 0
    while trim_ms < len(sound):
        if sound[trim_ms:trim_ms + chunk_size].dBFS > silence_thresh:
            return trim_ms
        trim_ms += chunk_size
    return trim_ms

def trim_silence(audio, silence_thresh=-40, chunk_size=10):
    start = detect_leading_silence(audio, silence_thresh, chunk_size)
    end = detect_leading_silence(audio.reverse(), silence_thresh, chunk_size)
    return audio[start:len(audio) - end]

def clean_sentence(text: str) -> str:
    text = text.strip()
    if text in {'"', "'", '.', ',', '."', "'"}:
        return ''
    text = re.sub(r'^[\'"]+', '', text)
    text = re.sub(r'[\'"]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

@lru_cache(maxsize=5)
def load_translation_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def translate_text_huggingface(text, src_lang='en', tgt_lang='es'):
    model, tokenizer = load_translation_model(src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    translated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def detect_emotion(text):
    try:
        label = emotion_classifier(text)[0][0]["label"].lower()
        if label in ["joy", "happiness"]:
            return "happy"
        elif label in ["anger"]:
            return "angry"
        elif label in ["sadness", "fear", "disgust"]:
            return "sad"
        else:
            return "neutral"
    except:
        return "neutral"

def split_into_narration_and_dialogues(text):
    pattern = r'"(.*?)"'
    dialogues = re.findall(pattern, text)
    parts = re.split(pattern, text)
    result = []
    for i, part in enumerate(parts):
        if i % 2 == 0 and part.strip():
            result.append({"type": "narration", "text": part.strip()})
        elif i % 2 == 1:
            result.append({"type": "dialogue", "text": part.strip()})
    return result

# ---------- Audio Generation ----------

def generate_narration_only_audio(text, speed, language, speaker_path, emotion, prompt, speaker_display_name):
    if language != "en":
        text = translate_text_huggingface(text, src_lang="en", tgt_lang=language)
    cleaned_text = clean_sentence(text)
    if not cleaned_text:
        return b'', ''
    final_path = sanitize_filename(prompt, speaker_display_name)
    with tts_lock:
        tts.tts_to_file(
            text=cleaned_text,
            speaker_wav=speaker_path,
            language=language,
            emotion=emotion,
            speed=speed,
            file_path=final_path
        )
    with open(final_path, "rb") as f:
        return f.read(), final_path

def generate_narration_with_dialogue_audio(segments, emotion, speed, language, narrator_voice_path, dialogue_voice_path, prompt, speaker_display_name):
    combined = AudioSegment.silent(duration=500)
    for segment in segments:
        text = clean_sentence(segment["text"])
        if not text:
            continue
        if language != "en":
            text = translate_text_huggingface(text, src_lang="en", tgt_lang=language)
        speaker_path = narrator_voice_path if segment["type"] == "narration" else dialogue_voice_path
        segment_emotion = emotion if segment["type"] == "narration" else detect_emotion(text)
        temp_filename = f"temp_{uuid.uuid4().hex}.wav"
        with tts_lock:
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_path,
                language=language,
                emotion=segment_emotion,
                speed=speed,
                file_path=temp_filename
            )
        audio = AudioSegment.from_wav(temp_filename)
        audio = trim_silence(audio).fade_in(20).fade_out(20)
        combined += audio + AudioSegment.silent(duration=300)
        os.remove(temp_filename)
    final_path = sanitize_filename(prompt, speaker_display_name)
    combined.export(final_path, format="wav")
    with open(final_path, "rb") as f:
        return f.read(), final_path
    
# ---------- LLM Handling ----------

def get_prompt(split_voices: bool, level: str) -> str:
    return {
        False: {
            "short": SHORT_NARRATION_PROMPT,
            "medium": MEDIUM_NARRATION_PROMPT,
            "long": LONG_NARRATION_PROMPT
        },
        True: {
            "short": SHORT_DIALOGUE_PROMPT,
            "medium": MEDIUM_DIALOGUE_PROMPT,
            "long": LONG_DIALOGUE_PROMPT
        }
    }[split_voices][level]

def get_llama3_response(user_input, split_voices):
    chat_history.append({"role": "user", "content": user_input})
    if "[PARA_LEVEL:1–3]" in user_input:
        model_name = "llama3.2:1b"
        num_predict = 2000
        level = "short"
    elif "[PARA_LEVEL:4–7]" in user_input:
        model_name = "mistral:7b-instruct"
        num_predict = 2000
        level = "medium"
    else:
        model_name = "llama3"
        num_predict = 2000
        level = "long"
    prompt = get_prompt(split_voices, level)
    stripped_input = re.sub(r'\[PARA_LEVEL:.*?\]', '', user_input).strip()
    full_prompt = f"{prompt}{stripped_input}"
    chat_history[-1]["content"] = full_prompt
    response = ollama.chat(
        model=model_name,
        messages=chat_history,
        options={
            "num_predict": num_predict,
            "temperature": 0.9,
            "top_p": 0.95,
            "stop": []
        }
    )
    reply = response.message.content
    chat_history.append({"role": "assistant", "content": reply})
    return reply

# ---------- gRPC Service ----------

class StoryServiceServicer(story_service_pb2_grpc.StoryServiceServicer):
    def GenerateStory(self, request, context):
        try:
            prompt = request.prompt
            emotion = request.emotion
            speed = request.speed
            language = request.language
            speaker_audio_path = request.speaker_audio
            split_voices = request.include_narration

            speaker_display_name = os.path.basename(speaker_audio_path).split(".")[0].title()
            story_text = get_llama3_response(prompt, split_voices)

            if split_voices:
                segments = split_into_narration_and_dialogues(story_text)
                dialogue_voice = "voices/female.wav"
                audio_data, _ = generate_narration_with_dialogue_audio(
                    segments=segments,
                    emotion=emotion,
                    speed=speed,
                    language=language,
                    narrator_voice_path=speaker_audio_path,
                    dialogue_voice_path=dialogue_voice,
                    prompt=prompt,
                    speaker_display_name=speaker_display_name
                )
            else:
                audio_data, _ = generate_narration_only_audio(
                    text=story_text,
                    speed=speed,
                    language=language,
                    speaker_path=speaker_audio_path,
                    emotion=emotion,
                    prompt=prompt,
                    speaker_display_name=speaker_display_name
                )

            return story_service_pb2.StoryResponse(
                audio=audio_data,
                text=story_text,
                message="success"
            )
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return story_service_pb2.StoryResponse(audio=b'', text='', message="error")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    story_service_pb2_grpc.add_StoryServiceServicer_to_server(StoryServiceServicer(), server)
    print("🚀 Starting gRPC server on port 50051...")
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()