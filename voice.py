import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import requests
import json
import cv2
import pytesseract

def recognize_objects(image_path):
    # Use OpenCV for object recognition
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)

def detect_currency(image_path):
    # Use pytesseract for currency detection
    text = pytesseract.image_to_string(image_path)
    return "Currency detected: " + text

def listen():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand what you said."
    except sr.RequestError as e:
        return "Sorry, I couldn't request results; {0}".format(e)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    speak("Hello! I am your voice assistant. How can I assist you today?")
    
    while True:
        user_input = listen()
        
        if "exit" in user_input:
            speak("Goodbye!")
            break
        elif "path" in user_input:
            # Assuming you have a file named 'image.jpg' in the same directory
            object_count = recognize_objects("image.jpg")
            speak("Number of objects detected: " + str(object_count))
        elif "currency" in user_input:
            currency_text = detect_currency("currency_image.jpg")
            speak(currency_text)
        elif "time" in user_input:
            current_time = datetime.datetime.now().strftime("%H:%M")
            speak("The current time is " + current_time)
        elif "weather" in user_input:
            # Assuming you have an API key and a weather API endpoint
            api_key = "YOUR_API_KEY"
            url = "http://api.openweathermap.org/data/2.5/weather?q=CityName&appid=" + api_key
            response = requests.get(url)
            weather_data = json.loads(response.text)
            temperature = round(weather_data["main"]["temp"] - 273.15, 2)
            description = weather_data["weather"][0]["description"]
            speak("The weather is " + description + " with a temperature of " + str(temperature) + " degrees Celsius.")
        elif "youtube" in user_input:
            webbrowser.open("https://www.youtube.com")
        elif "chrome" in user_input:
            webbrowser.open("https://www.google.com")
        else:
            speak("I'm sorry, I didn't understand that. Can you please repeat?")

if __name__ == "__main__":
    main()
