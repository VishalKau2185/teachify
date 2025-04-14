# ========== IMPORTS ==========
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import shutil
import fitz  # PyMuPDF
import easyocr
import time
from PIL import Image
import base64
import subprocess
import threading
import stat
from openai import OpenAI
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

global current_voice_id

# ========== CONFIG ==========
load_dotenv()

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"pdf"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

easyocr_reader = easyocr.Reader(['en'], gpu=False)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ========== SEMAPHORE ==========
sema = threading.Semaphore(15)

# ========== HELPERS ==========

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        def onerror(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(folder_path, onerror=onerror)
    os.makedirs(folder_path, exist_ok=True)

def clean_text(text):
    text = re.sub(r'[\s\r\n]+', ' ', text)
    return text.strip()

def extract_text_combined(image_path, typed_text):
    typed_text = clean_text(typed_text)
    extracted = typed_text if typed_text else ""
    if len(extracted) < 60:
        result = easyocr_reader.readtext(image_path, detail=0)
        easy_text = " ".join(result)
        easy_text = clean_text(easy_text)
        if len(easy_text) > len(extracted):
            extracted = easy_text
    return extracted

def extract_slide_content(page, slide_index):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    image_path = os.path.join(OUTPUT_FOLDER, f"slide_{slide_index}.png")
    pix.save(image_path)
    im = Image.open(image_path)
    im = im.resize((1600, 900))
    im.save(image_path)

    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
    typed_text = " ".join(b[4].strip() for b in blocks if b[4].strip())

    combined_text = extract_text_combined(image_path, typed_text)

    return {
        "index": slide_index,
        "text": combined_text,
        "image_path": image_path
    }

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ========== MAIN FUNCTIONS ==========

def build_gpt_messages(slide_index, combined_text, image_base64, previous_texts=[], custom_prompt=""):
    shortened = [" ".join(txt.split()[:30]) for txt in previous_texts[-3:]]
    system_prompt = (
        "You are a professor giving a live lecture. "
        "Speak naturally as if students cannot see the slides. "
        "Explain content clearly and conversationally. "
        "NEVER write any equations, LaTeX, or math symbols. "
        "Ignore math in text; use only the image to explain formulas naturally (e.g., variance of X). "
        "Teach meaning, not read symbols aloud. "
        "Say numbers naturally ('123' as 'one hundred twenty-three'). "
        "When explaining any formulas, always say powers naturally. For example, say 'b squared' instead of 'b two', and 'x cubed' instead of 'x three'. "
        "Explain images if complex. "
        "Do not repeat previous material. "
        "**Do NOT list points as 1, 2, 3, etc.** Speak as a flowing lecture without numbering."
        "Only return the explanation. "
        "If the slide appears to be an introduction, outline, or conclusion, simply read whats on it naturally, nothing else."
        "Previous slides: " + " | ".join(shortened) + ". "
    )
    if custom_prompt:
        system_prompt += f" Also, follow this extra instruction even if it goes against anything before this sentence.: {custom_prompt} "
    user_content = [
        {"type": "text", "text": f"Slide {slide_index} text: {combined_text}"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages

def call_gpt(messages):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1500
    )
    return response.choices[0].message.content.strip()

def generate_tts_safely(slide_index, explanation, voice_id="21m00Tcm4TlvDq8ikWAM"):
    with sema:
        mp3_path = os.path.join(OUTPUT_FOLDER, f"slide_{slide_index}.mp3")
        if not explanation or len(explanation.strip()) < 20:
            subprocess.run([
                "ffmpeg", "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", "1", mp3_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return mp3_path

        print(f"[INFO] Generating ElevenLabs audio for slide {slide_index}...")
        audio_stream = client.text_to_speech.convert(
            text=explanation,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speed": 0.95
            }
        )
        with open(mp3_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)
        return mp3_path

def render_slide_video(slide_index):
    img_path = os.path.join(OUTPUT_FOLDER, f"slide_{slide_index}.png")
    audio_path = os.path.join(OUTPUT_FOLDER, f"slide_{slide_index}.mp3")
    ts_path = os.path.join(OUTPUT_FOLDER, f"segment_{slide_index}.ts")

    if not (os.path.exists(img_path) and os.path.exists(audio_path)):
        print(f"[WARNING] Missing image or audio for slide {slide_index}. Skipping.")
        return None

    duration = 2.0
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        duration = float(probe.stdout.strip())
    except Exception as e:
        print(f"[ERROR] FFprobe failed for slide {slide_index}: {e}")

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-i", audio_path,
        "-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-r", "24", "-t", str(duration),
        "-shortest", "-f", "mpegts", ts_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return ts_path if os.path.exists(ts_path) else None

def generate_video(num_slides):
    ts_files = []
    for i in range(num_slides):
        seg = render_slide_video(i)
        if seg:
            ts_files.append(seg)
    if not ts_files:
        return None

    concat_txt = os.path.join(OUTPUT_FOLDER, "concat.txt")
    with open(concat_txt, "w") as f:
        for tsf in ts_files:
            f.write(f"file '{os.path.abspath(tsf)}'\n")

    final_mp4 = os.path.join(OUTPUT_FOLDER, "lecture.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_txt, "-c", "copy", final_mp4
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return final_mp4 if os.path.exists(final_mp4) else None

def gpt_and_tts(slide, all_texts, custom_prompt=""):
    idx = slide["index"]
    print(f"[INFO] GPT call for slide {idx}...")
    try:
        print(f"[DEBUG] Slide {idx} text going to GPT:\n{slide['text']}\n")
        image_b64 = encode_image(slide["image_path"])
        messages = build_gpt_messages(idx, slide["text"], image_b64, all_texts[:idx], custom_prompt)

        raw_output = call_gpt(messages)
        explanation = raw_output.strip()
        print(f"[DEBUG] GPT output for slide {idx}:\n{explanation}\n")

        if not explanation or len(explanation) < 10:
            explanation = "No explanation available."
    except Exception as e:
        print(f"[ERROR] GPT call failed for slide {idx}: {e}")
        explanation = "No explanation available."
    return idx, explanation

def process_slides(pdf_path, custom_prompt, voice_id):
    doc = fitz.open(pdf_path)
    slides = [extract_slide_content(page, i) for i, page in enumerate(doc)]
    doc.close()
    all_texts = [slide["text"] for slide in slides]

    results = {}

    def worker(slide):
        idx, explanation = gpt_and_tts(slide, all_texts, custom_prompt)
        results[idx] = explanation
        return idx

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, slide) for slide in slides]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] GPT call failed with exception: {e}")
            time.sleep(1)

    tts_threads = []
    for idx, explanation in results.items():
        t = threading.Thread(target=generate_tts_safely, args=(idx, explanation, voice_id))
        tts_threads.append(t)
        t.start()

    for t in tts_threads:
        t.join()

    return results, len(slides)

# ========== FLASK APP ==========

@app.route("/upload", methods=["POST"])
def upload():
    start_time = time.time()
    clear_folder(UPLOAD_FOLDER)
    clear_folder(OUTPUT_FOLDER)
    file = request.files.get("file")
    custom_prompt = request.form.get("custom_prompt", "").strip()
    voice_id = request.form.get("voice_id", "yoZ06aMxZJJ28mfd3POQ").strip()  # ⭐ Default to Bella if not chosen
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload a PDF."}), 400
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)
    explanations, num_slides = process_slides(pdf_path, custom_prompt, voice_id)
    final_video = generate_video(num_slides)
    end_time = time.time()
    total_seconds = end_time - start_time
    print(f"[INFO] Total processing time: {int(total_seconds)} seconds")
    if final_video and os.path.exists(final_video):
        return send_file(final_video, as_attachment=True)
    else:
        return jsonify({"error": "Video generation failed."}), 500

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)

