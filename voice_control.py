import speech_recognition as sr

recognizer = sr.Recognizer()
voice_command = ""

def listen_command():
    global voice_command

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=2)
                command = recognizer.recognize_google(audio).lower()

                print("Heard:", command)

                valid = ["left", "right", "jump", "roll", "pause", "resume", "stop"]

                for v in valid:
                    if v in command:
                        voice_command = v
                        break

            except:
                pass