import streamlit as st
import grpc
import os
import json
import uuid
import time
import re
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment
from st_audiorec import st_audiorec
from proto import story_service_pb2
from proto import story_service_pb2_grpc

# Constants
GRPC_SERVER_ADDRESS = "localhost:50051"
voices_dir = "voices"
speakers_json = os.path.join(voices_dir, "speakers.json")
os.makedirs(voices_dir, exist_ok=True)

# Session state
st.session_state.setdefault("recording", False)
st.session_state.setdefault("audio_blob", None)
st.session_state.setdefault("audio_jobs", [])
st.session_state.setdefault("audio_results", [])
st.session_state.setdefault("spinner_triggered", False)

if "speaker_choices" not in st.session_state:
    if os.path.exists(speakers_json):
        with open(speakers_json, "r") as f:
            st.session_state.speaker_choices = json.load(f)
    else:
        st.session_state.speaker_choices = {"Default": "voices/Default Speaker.wav"}
        with open(speakers_json, "w") as f:
            json.dump(st.session_state.speaker_choices, f, indent=4)

# gRPC setup
channel = grpc.insecure_channel(GRPC_SERVER_ADDRESS, options=[
    ('grpc.max_send_message_length', 100 * 1024 * 1024),
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
])
stub = story_service_pb2_grpc.StoryServiceStub(channel)
executor = ThreadPoolExecutor(max_workers=3)

def generate_audio(prompt, emotion, speed, language, speaker_audio, split_voices):
    request = story_service_pb2.StoryRequest(
        prompt=prompt, emotion=emotion, speed=speed,
        language=language, speaker_audio=speaker_audio,
        include_narration=split_voices
    )
    response = stub.GenerateStory(request)
    if response.audio:
        audio_id = uuid.uuid4().hex
        audio_filepath = f"generated_audio_{audio_id}.wav"
        with open(audio_filepath, "wb") as f:
            f.write(response.audio)
        return audio_filepath, response.text, prompt
    return None, None, prompt

def handle_voice_upload_bytes(audio_bytes, speaker_name):
    if len(audio_bytes) < 15000:
        st.error("Audio too short — must be at least 15 seconds.")
        return False
    if speaker_name in st.session_state.speaker_choices:
        st.error(f"A speaker named '{speaker_name}' already exists.")
        return False
    speaker_path = os.path.join(voices_dir, f"{speaker_name}.wav")
    with open(speaker_path, "wb") as f:
        f.write(audio_bytes)
    st.session_state.speaker_choices[speaker_name] = speaker_path
    with open(speakers_json, "w") as f:
        json.dump(st.session_state.speaker_choices, f, indent=4)
    st.success(f"'{speaker_name}' added successfully!")
    return True

# UI
st.set_page_config(page_title="Story2Audio (Concurrent)", layout="centered")
st.title("📖 Story2Audio Generator 🎤")

with st.form("audio_form"):
    prompt = st.text_area("Story Prompt", height=80)
    para_choice = st.selectbox("Story Length", ["1–3", "4–7", "8+"])
    speaker_name = st.selectbox("Speaker Voice", list(st.session_state.speaker_choices.keys()))
    language = st.selectbox("Language", ["en", "es", "fr", "de", "hi", "it", "ru"])
    voice_mode = st.selectbox("Voice Mode", ["Narration Only", "Narration + Dialogue"])
    emotion = st.selectbox("Emotion", ["neutral", "happy", "sad", "angry"])
    speed = st.slider("Speech Speed", 0.7, 1.3, 1.0, 0.05)

    uploaded_voice = st.file_uploader("Upload Voice", type=["wav"])
    upload_btn = st.form_submit_button("Upload File")
    record_name = st.text_input("Recording Name (optional)")
    record_btn = st.form_submit_button("Record Voice")
    generate_btn = st.form_submit_button("Generate Audio")

if upload_btn and uploaded_voice:
    handle_voice_upload_bytes(uploaded_voice.read(), os.path.splitext(uploaded_voice.name)[0])

if record_btn:
    st.session_state.recording = True
    st.session_state.audio_blob = None

if st.session_state.recording:
    audio_bytes = st_audiorec()
    if audio_bytes:
        st.session_state.audio_blob = audio_bytes
        st.success("Recording captured!")
    if st.button("Upload Recording"):
        if not record_name.strip():
            st.warning("Please enter a recording name.")
        elif handle_voice_upload_bytes(st.session_state.audio_blob, record_name.strip()):
            st.session_state.recording = False
    if st.button("Cancel"):
        st.session_state.recording = False

# Generate
if generate_btn and prompt.strip():
    full_prompt = f"[PARA_LEVEL:{para_choice}]\n\n{prompt}"
    job = executor.submit(
        generate_audio,
        full_prompt, emotion, speed, language,
        st.session_state.speaker_choices[speaker_name],
        voice_mode == "Narration + Dialogue"
    )
    st.session_state.audio_jobs.append((job, str(uuid.uuid4()), full_prompt))
    st.session_state.spinner_triggered = True

# Process jobs
pending_jobs = []
for job, uid, stored_prompt in st.session_state.audio_jobs:
    if job.done():
        audio_path, story_text, _ = job.result()
        if audio_path:
            st.session_state.audio_results.append((audio_path, story_text, stored_prompt))
    else:
        pending_jobs.append((job, uid, stored_prompt))

st.session_state.audio_jobs = pending_jobs

# Show running prompts (cleaned)
if pending_jobs:
    st.info("Generating audio for:")
    for _, _, prompt_text in pending_jobs:
        cleaned = re.sub(r'\[PARA_LEVEL:.*?\]', '', prompt_text).strip()
        st.markdown(f"• 🛠️ `{cleaned[:60]}...`")

for audio_path, story_text, original_prompt in st.session_state.audio_results:
    clean_prompt = re.sub(r'\[PARA_LEVEL:.*?\]', '', original_prompt).strip()
    short_prompt = clean_prompt[:60] + "..." if len(clean_prompt) > 60 else clean_prompt
    st.subheader(f"{short_prompt}")
    st.audio(audio_path)
    with st.expander("Story Text"):
        st.write(story_text)

# Auto-rerun
if pending_jobs:
    time.sleep(2)
    st.rerun()
