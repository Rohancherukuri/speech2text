# Importing necessary modules
import time
# from speech_application import SpeechTransformerUI
from s2t import *
# from speech_models import SpeechClassifier, SentimentSpeechAnalyzer

def main():
    """This is a main function"""
    try:
        
        params = {
                    "processor": AudioProcessor(
                                                    audio_path="./Sample_Audio_Files/audio4.wav",
                                                    audio_format="wav", 
                                                    to_path="./Transformed_Audio_Files/t_rec4.wav",
                                                    boost=12
                        
                                                ),
                    "transformer": SpeechTransformer(
                                                            
                                                audio_path="./Transformed_Audio_Files/t_rec4.wav",
                                                model_size="large",
                                                task="translate"
                                                    ),
            
                    "classifier": SpeakerClassifier(
                                                        audio_path="./Transformed_Audio_Files/t_rec4.wav",
                                                        num_speakers=2
                                                    )
                 }
        
        stp = SpeechTransformerPipeline(**params)
        print(f"The Speech To Text Pipeline Model is: {stp}")
        t1 = time.perf_counter()
        stp.process_audio()

        
        stp.complete_transform(text_file_translated="./report/text_translated.txt", 
                                         clf_report_translated="./report/speech_translated.xlsx")
        t2 = time.perf_counter()
        print(f"[Finished in: {round(t2 - t1, 3)} sec(s)]")
    except Exception as e:
        print("Error occured in main function! " + str(e))


if __name__ == "__main__":
    main()
