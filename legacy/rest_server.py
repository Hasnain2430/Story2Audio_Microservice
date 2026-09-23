from flask import Flask, request, jsonify
import grpc
import base64
import os

from proto import story_service_pb2
from proto import story_service_pb2_grpc

app = Flask(__name__)
GRPC_SERVER_ADDRESS = "localhost:50051"

@app.route("/generate-story/", methods=["POST"])
def generate_story():
    try:
        data = request.get_json()
        prompt = data["prompt"]
        emotion = data["emotion"]
        speed = float(data["speed"])
        language = data["language"]
        include_narration = bool(data["include_narration"])
        speaker_audio_b64 = data["speaker_audio_base64"]

        # Save base64 audio to file
        speaker_path = "uploaded_speaker.wav"
        with open(speaker_path, "wb") as f:
            f.write(base64.b64decode(speaker_audio_b64))

        # Setup gRPC
        channel = grpc.insecure_channel(GRPC_SERVER_ADDRESS)
        stub = story_service_pb2_grpc.StoryServiceStub(channel)

        grpc_request = story_service_pb2.StoryRequest(
            prompt=prompt,
            emotion=emotion,
            speed=speed,
            language=language,
            speaker_audio=speaker_path,
            include_narration=include_narration
        )

        response = stub.GenerateStory(grpc_request)

        with open("response_audio.wav", "wb") as f:
            f.write(response.audio)

        return jsonify({
            "text": response.text,
            "message": response.message,
            "audio_file": "response_audio.wav"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=8000)
