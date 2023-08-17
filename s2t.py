# Multilingual Speech Recognition using OpenAI Whisper model
import os
import wave
import time
import flet
import torch
import joblib
import whisper
import datetime
import contextlib
import numpy as np
import pandas as pd
from shutup import please
from pydub import AudioSegment
from typing import Optional, Any
from pyannote.core import Segment
from pyannote.audio import Audio
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

# SentimentSpeechAnalyzer class
class SentimentSpeechAnalyzer(torch.nn.Module):
    """This is a SentimentSpeechAnlyzer class"""
    def __init__(self, text: str="") -> None:
        """This is a SentimentSpeechAnalyzer constructor"""
        self.text = text
        self.model = None
        super().__init__()
    
    def analyze(self) -> str:
        """This method analyzes the sentiment of the text"""
        try:
            # Importing neceesary modules
            from transformers import pipeline
            
            self.model = pipeline("sentiment-analysis")
            result = self.model(self.text)
            return result
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the analyze method of SentimentSpeechAnalyzer class: " + str(e))
    
    def save(self, file_path: str) -> None:
        """This method saves the SentimentSpeechAnalyzer model"""
        print("Saving the SentimentSpeechAnalyzer model...")
        torch.save(model, path=file_path)
    
    
    def load(self, file_name: str) -> Any:
        """This method loads the SentimentSpeechAnalyzer model"""
        print("Loading the SentimentSpeechAnalyzer model...")
        model = torch.load(path=file_path)
        return model
    
    
    def __repr__(self) -> str:
        """This is a special method"""
        return f"SentimentSpeechAnalyzer(text={self.text})"
    
# SpeakerClassifier class
class SpeakerClassifier(torch.nn.Module):
    """This is a SpeakerClassifier class"""
    def __init__(self, audio_path: str="", num_speakers: int=0) -> None:
        """This is a SpeakerClassifier constructor"""
        self.audio_path = audio_path
        self.num_speakers = num_speakers
        super().__init__()
    
    def classify(self, segments: list) -> Any:
        """This method classifies the Audio recordings"""
        try:
            # Importing necessary modules
            please()
            embedding_model = PretrainedSpeakerEmbedding( 
                                                "speechbrain/spkrec-ecapa-voxceleb",
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                            )
            _, file_ending = os.path.splitext(f"{self.audio_path}")
            audio_file = self.audio_path.replace(file_ending, ".wav")
            def convert_time(secs: int | float) -> datetime.timedelta:
                """This function converts the seconds to timedelta"""
                return datetime.timedelta(seconds=round(secs))
            
            
            def embedding(segment: Segment) -> pd.DataFrame:
                """This is an inner embedding function"""
                # Get duration
                with contextlib.closing(wave.open(self.audio_path)) as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
                audio = Audio()
                # Whisper overshoots the end timestamp in the last segment
                start = segment["start"]
                end = min(duration, segment["end"])
                clip = Segment(start, end)
                waveform, sample_rate = audio.crop(audio_file, clip)
                return embedding_model(waveform[None])
            
            embeddings = np.zeros(shape=(len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = embedding(segment)
            embeddings = np.nan_to_num(embeddings)
            # Assign speaker label
            if self.num_speakers == 0:
                score_num_speakers = {}
                for self.num_speakers in range(2, 11):
                    clustering = AgglomerativeClustering(self.num_speakers).fit(embeddings)
                    score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                    score_num_speakers[self.num_speakers] = score
                best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
            else:
                best_num_speaker = self.num_speakers
            clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
            labels = clustering.labels_
            for i in range(len(segments)):
                segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)
            # Make output
            objects = {
                "Start" : [],
                "End": [],
                "Speaker": [],
                "Text": []
            }
            text = ""
            for (i, segment) in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    objects["Start"].append(str(convert_time(segment["start"])))
                    objects["Speaker"].append(segment["speaker"])
                    if i != 0:
                        objects["End"].append(str(convert_time(segments[i - 1]["end"])))
                        objects["Text"].append(text)
                        text = ""
                text += segment["text"] + " "
            objects["End"].append(str(convert_time(segments[i - 1]["end"])))
            objects["Text"].append(text)
            return pd.DataFrame(objects)
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the classify method of the SpeakerClassifier class " + str(e))
    
    def save(self, file_path: str) -> None:
        """This method saves the SpeakerClassifier model"""
        print("Saving the SpeakerClassifier model...")
        torch.save(model, path=file_path)
    
    
    def load(self, file_name: str) -> None:
        """This method loads the SpeakerClassifier model"""
        print("Loading the SpeakerClassifier model...")
        model = torch.load(path=file_path)
        return model
    
    def __repr__(self) -> str:
        """This is a special method"""
        return f"SpeakerClassifier(audio_path={self.audio_path})"

# AudioProcessor class
class AudioProcessor(torch.nn.Module):
    """This is a AudioProcessor class"""
    def __init__(self, audio_path: str="", audio_format: str="wav", to_path: str="", boost: int=1) -> None:
        """This is a AudioProcessor constructor"""
        self.audio_path = audio_path
        self.audio_format = audio_format
        self.to_path = to_path
        self.boost = boost
        super().__init__()

    def process(self) -> None:
        """This method is used for cleaning audio files"""
        try:
            # Loading the audio file
            audio = AudioSegment.from_file(self.audio_path, format=self.audio_format)
            # Testing on first 3min o the audio file
            first_3min = 180 * 1000 # converting milli seconds to minutes
            audio = audio[:first_3min]
            # Increasing the audio volume
            audio += self.boost
            # Exporting to the cleaned audio file to other directory
            audio.export(self.to_path, format=self.audio_format)
        except Exception as e:
            print("An error occured in process method of AudiopProcessor class")
    
    def save(self, file_path: str) -> None:
        """This method saves the AudioProcessor model"""
        print("Saving the AudioProcessor model...")
        torch.save(model, path=file_path)
    
    
    def load(self, file_name: str) -> Any:
        """This method loads the AudioProcessor model"""
        print("Loading the AudioProcessor model...")
        model = torch.load(path=file_path)
        return model
    
    def __repr__(self) -> str:
        """This is a special method"""
        return f"AudiopProcessor(audio_path={self.audio_path}, audio_format={self.audio_format}, to_path={self.to_path})"


# SpeechTransformer class    
class SpeechTransformer(torch.nn.Module):
    """This is a SpeechTransformer class"""
    def __init__(self, audio_path: str="", model_size: str="small", task: str="transcribe") -> None:
        """This is constructor for SpeechTransformer"""
        """You can set the model_size to be ["tiny", small", "base", "medium", "large", "large-v1", "large-v2"]"""
        self.audio_path = audio_path
        self.model_size = model_size
        self.task = task
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__()
    
    def get_data(self) -> dict:
        """This method returns the detected language and the text of the audio recording"""
        try:
            # Removing warnings on the screen
            please()
            # Loading the model
            if self.task not in ["transcribe", "translate"]:
                raise ValueError(f"Task {self.task} is not present in the given actionable tasks.")
                
            else:
                self.model = whisper.load_model(self.model_size, device=self.device)
                options=dict(fp16=False, beam_size=5, best_of=5, task=self.task)
                result = self.model.transcribe(self.audio_path, **options)

                # Returning the resultant transcribed text
                segments = []
                i = 0
                for segment in result["segments"]:
                    chunk = {}
                    chunk["start"] = segment["start"]
                    chunk["end"] = segment["end"]
                    chunk["text"] = segment["text"]
                    segments.append(chunk)
                    i += 1
                # Returning the resultant transcribed text
                return {"Text": result["text"], "Segments": segments}
        except (KeyboardInterrupt, Exception) as e:
            print("An error occured at get_text method of SpeechTransformer class " + str(e))
    
    def save(self, file_path: str) -> None:
        """This method saves the SpeechTransformer model"""
        print("Saving the SpeechTransformer model...")
        torch.save(self.model, path=file_path)
    
    def load(self, file_name: str) -> Any:
        """This method loads the SpeechTransformer model"""
        print("Loading the SpeechTransformer model...")
        model = torch.load(path=file_path)
        return model
    
    def __repr__(self) -> str:
        """This is a special method"""
        return f"SpeechTransformer(audio_path={self.audio_path}, model_size={self.model_size})"

        
    
class SpeechTransformerPipeline(torch.nn.Module):
    """This is a SpeechTransformerPipeline class"""
    def __init__(self, **kwargs) -> None:
        """This is a SpeechTransformerPipeline constructor"""
        self.memory = kwargs.copy()
        self.shelf = {key: type(val) for key, val in kwargs.items()}
        super().__init__()
    
    def process_audio(self) -> None:
        """This method is used for preprocessing the audio"""
        try:
            self.audio_processor = self.memory.get("processor")
            self.audio_processor.process()
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the process_audio method of SpeechTransformerPipeline class! " + str(e))
            
                       
    def complete_transform(self, text_file_translated: str, clf_report_translated: str) -> str:
        """This method is used for classifying the audio which is given as an input"""
        try:
            # Setting maximum column width
            pd.set_option("display.max_colwidth", 0)
            speech_model = self.memory.get("transformer")
            speech_model.task = "translate"
            self.result = speech_model.get_data()
            classifier = self.memory.get("classifier")
            classified_text_data = classifier.classify(self.result["Segments"])
            self.text_data = classified_text_data
            self.write(text_file_translated)
            self.text_data.to_excel(clf_report_translated, index=False)
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the complete_transform_translated method of SpeechTransformerPipeline class " + str(e))
    
           
    def analyze_audio(self) -> str:
        """This method is used for analyzing the sentiment of the text from audio"""
        try:
            self.analyzer = self.memory.get("analyzer")
            result = self.analyzer.analyze(self.text)
            return result
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the analyze_audio method of SpeechTransformerPipeline class! " + str(e))
    
    def write(self, filename: str) -> None:
        """This method saves the text in file format"""
        try:
            with open(filename, "w") as file:
                file.write(self.result["Text"])
            print("Saved the text!")
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside write method of SpeechTransformerPipeline class! " + str(e))
    
    def read(self, filename: str) -> str:
        """This method read's the text file which is saved in file format"""
        try:
            with open(filename, "r") as file:
                content = file.read()
            return content
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the read method of SpeechTransformerPipeline class " + str(e))    
            
    def save(self, file_path: str) -> None:
        """Save the SpeechTransformerPipeline model"""
        try:
            print("Saving the SpeechTransformerPipeline model...")
            torch.save(model, path=file_path)
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the save method of SpeechTransformerPipeline class! " + str(e))
    
    def load(self, file_name: str) -> Any:
        """Load the SpeechTransformerPipeline model"""
        try:
            print("Loading the SpeechTransformerPipeline model...")
            model = torch.load(path=file_path)
            return model
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the load method of the SpeechTransformerPipeline class! " + str(e))
    
    
    def get_requirements(self) -> str:
        """This method create's a requirements.txt file for the SpeechTransformerPipeline"""
        try:
            # Importing the os module
            import os
            
            flag = os.system("pip freeze > requirements.txt")
            if flag == 0:
                return "Successfull!"
            else:
                return "Unsuccessfull!"
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the get_requirements method of the SpeechTransformerPipeline class! " + str(e))
    
    def dockerize(self, container_name: str, model_path: str, dir_path: str, file_name: str) -> None:
        """This method is used for dockerizing the SpeechTransformerPipeline model"""
        try:
            simplefilter("ignore")
            print("Checking for docker...")
            flag = os.system("docker --version")
            if flag != 0:
                print("There is no docker present in your machine!")
                print(f"Exitted with flag {flag}!")
            self.cn = container_name
            dockerfile = os.path.isfile("Dockerfile")
            if dockerfile is True:
                cmd = ["FROM python:3.8 \n\n", 
                       "WORKDIR /modeldir \n\n", 
                       "COPY requirements.txt requirements.txt \n\n",
                       "RUN pip install -r requirements.txt \n\n",
                       "ADD . . \n\n",
                       f"COPY {model_path} ./modeldir \n\n",
                       f"""CMD ["python", {file_name}]"""]
                with open("Dockerfile", "w") as f:
                    f.writelines(cmd)
                print("Dockerizing the SpeechTransformerPipeline...")
                os.system(f"docker build  -f {dir_path}/Dockerfile -t {container_name} . ")
            else:
                PATH = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/datascience-ml-dev3/code/Users/datascience-aml/Rohan/speech2text"
                print("Dockerizing the SpeechTransformerPipeline...")
                os.system(f"docker build  -f {PATH}/Dockerfile -t {container_name} . ")
                print("Completed building the docker image!")
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured: inside the dockerize method from the SpeechTransformerPipeline class " + str(e))
            
    
    def run_docker(self) -> None:
        """This method run's the docker container of the SpeechTransformerPipeline"""
        try:
            print("Running the created docker image...")
            flag = os.system(f"docker run -it {self.cn}")
            if flag == 0:
                print("Successfully ran the docker image!")
            else:
                print("Unscuccessfull in running the docker image!")
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the run_docker method from the SpeechTransformerPipeline class " + str(e))
    
    
    def serve(self, name: str="Speech2Text", host: str="192.168.18.88", port: int=5000) -> None:
        def launch(page: ft.Page) -> None:
            try:
                obj = ft.Container()
                page.add()
            except (KeyboardInterrupt, Exception) as e:
                print("Error occured inside the serve method of the SpeechTransformerPipeline class " + str(e))
        
        ft.flet.app(target=launch,)
    
    def get_method_names(self) -> list:
        """This method list's all the methods available in SpeechTransformerPipeline"""
        try:
            method_names = ["process_audio", "complete_transform_raw", 
                            "complete_transform_translated", "save", "load", "dockerize", 
                           "run_docker", "serve", "get_requirements", "analyze_audio"]
            
            return method_names
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the get_method_names of SpeechTransformerPipeline class! " + str(e))
    
    def __repr__(self) -> str:
        """This is a special method"""
        try:
            return f"SpeechTransformerPipeline({list(self.shelf.items())})"
        except (KeyboardInterrupt, Exception) as e:
            print("Error occured inside the __repr__ method of SpeechTransformerPipeline class! " + str(e))