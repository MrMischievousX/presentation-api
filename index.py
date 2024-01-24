# Importing Libraries

from flask import Flask, request, jsonify
import io
import warnings
import textract as tp
import moviepy.editor as mp
import whisper
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import cv2
import os
import shutil

# Setting environment
warnings.filterwarnings("ignore")

# Function to calculate text relevancy score

def getTextRelevancy(paragraph, topic):
  # create a dataset with the paragraph and the topic
  data = pd.DataFrame({'text': [paragraph, topic], 'label': [1, 0]})

  # create a TF-IDF vectorizer
  vectorizer = TfidfVectorizer()

  # transform the text data into a TF-IDF matrix
  tfidf_matrix = vectorizer.fit_transform(data['text'])

  # split the TF-IDF matrix back into the paragraph and the topic
  tfidf_paragraph = tfidf_matrix[0]
  tfidf_topic = tfidf_matrix[1]

  # train a logistic regression classifier on the dataset
  classifier = LogisticRegression()
  classifier.fit(tfidf_matrix, data['label'])

  # predict the relevancy of the paragraph to the topic
  relevancy_score = classifier.predict_proba(tfidf_paragraph.reshape(1, -1))[0][0]
  relevancy_score = (relevancy_score * 100)
  relevancy_score = relevancy_score - int(relevancy_score)

  # print the relevancy score
  print("Relevancy score :", relevancy_score)
  return relevancy_score

# Extract text from ppt and clean the data

def getPresentationContent(path):
  pptText = tp.process(path)
  pptText = pptText.decode('utf-8')
  pptText = pptText.strip()
  pptText = pptText.replace("\n\n"," ")
  return pptText

# Convert video to audio

def getAudioFromVideo(path):
  my_clip = mp.VideoFileClip(path)
  audio_path = os.path.join('uploads', "audio.wav")
  my_clip.audio.write_audiofile(audio_path)
  return audio_path

# Extract text from audio

def getTextFromAudio(path):
  model = whisper.load_model("base")
  result = model.transcribe(path, fp16=False)
  audioText = result["text"]
  return audioText

# Calculate audio confidence

def getConfidenceFromAudio(path):
  # Load the audio file
  sample_rate, audio_data = wav.read(path)

  # Extract audio features (e.g., MFCC coefficients)
  mfcc_features = mfcc(audio_data, samplerate=sample_rate,nfft=2048)

  # Calculate the confidence score
  confidence_score = np.mean(mfcc_features)  # Example: using mean of MFCC coefficients

  # Normalize the confidence score between 0 and 1
  confidence_score_normalized = (confidence_score - np.min(mfcc_features)) / (np.max(mfcc_features) - np.min(mfcc_features))

  audioScore = confidence_score_normalized * 100
  audioScore = audioScore - int(audioScore)

  return audioScore

# Calculate video confidence

def getConfidenceFromVideo(path):
  video = cv2.VideoCapture(path)

  first_frame = None
  total_motion_area = 0
  total_frame_area = 0
  frame_count = 0

  while True:
      check, frame = video.read()
      if not check:
          break

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      gray = cv2.GaussianBlur(gray, (21, 21), 0)

      if first_frame is None:
          first_frame = gray
          total_frame_area = frame.shape[0] * frame.shape[1]
          continue

      delta_frame = cv2.absdiff(first_frame, gray)
      threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]
      threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

      contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      for contour in contours:
          if cv2.contourArea(contour) < 1000:
              continue
          (x, y, w, h) = cv2.boundingRect(contour)
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
          total_motion_area += w * h

      key = cv2.waitKey(1)
      if key == ord('q'):
          break

      frame_count += 1

  average_motion_area = total_motion_area / frame_count
  motion_percentage = (average_motion_area / total_frame_area) * 100
  net_confidence_score = 100 - motion_percentage
  net_confidence_score = max(0, net_confidence_score)
  videoScore = (net_confidence_score/100)

  return videoScore

# Initiating flask api

app=Flask(__name__)

path_map = {}

@app.route("/", methods=["GET"])
def home():
  return "STARTED"

@app.route("/upload", methods=["POST"])
def hello():
  os.makedirs('uploads', exist_ok=True)
  if 'pptx' not in request.files and 'video' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

  pptx_file = request.files.get('pptx')
  video_file = request.files.get('video')

  path_map["pptx"] = os.path.join('uploads', pptx_file.filename)
  pptx_file.save(path_map["pptx"])

  path_map["video"] = os.path.join('uploads', video_file.filename)
  video_file.save(path_map["video"])

  pptText = getPresentationContent(path_map["pptx"])

  path_map["audio"] = getAudioFromVideo(path_map["video"])

  audioText = getTextFromAudio(path_map["audio"])

  pptScore = getTextRelevancy(pptText, "presentation")
  audioTextScore = getTextRelevancy(audioText,"presentation")
  audioScore = getConfidenceFromAudio(path_map["audio"])
  videoScore = getConfidenceFromVideo(path_map["video"])

  finalScore = pptScore * 0.20 + audioTextScore * 0.35 + audioScore * 0.25 + videoScore * 0.20

  shutil.rmtree('uploads')

  return f"Final Score {finalScore:.2f}%", 200


