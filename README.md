Handwritten Digit Recognizer

A small web app that reads handwritten digits. You can draw one digit on a canvas, or upload a photo of a page with several digits and the app will read them in order.

The model is a neural network trained on the MNIST dataset using TensorFlow and Keras. The web page is made with Streamlit, and the predictions come from a FastAPI server running in the background.

What it does

- Draw a digit - Draw with the mouse, click Predict, and you get the digit, a confidence percentage and a bar chart for all ten digits.
- Upload a photo - The app finds each digit in the picture, draws a box around it, and reads them left to right, top to bottom. Green boxes mean the model is fairly sure. Orange boxes mean it is not.
- Works on lined paper - The ruled lines of a notebook are removed before the digits are read, so you can write on normal notebook pages.

How it works

There are two parts.

1. `backend.py` is the FastAPI server. It loads the trained model, cleans up the images and returns the predictions.
2. `app.py` is the Streamlit page. It shows the canvas and the upload box, sends the image to the server and shows the answer.

Before a digit goes into the model, it is cropped, scaled to fit a 20x20 box and centered inside a 28x28 image. This is how the MNIST images were made, and doing the same thing makes the predictions a lot better on real handwriting.

For photos with many digits, the server first turns the picture into black and white, removes the long thin lines of the paper, and then finds each separate shape. Shapes that are too small, too long or too close to the edge of the picture are thrown away. What is left is sorted into rows and read one by one.

Project files

app.py              Streamlit web page
backend.py          FastAPI server and image processing
digit_model.keras   Trained model
requirements.txt    Python packages
README.md

Setup

You need Python 3.10, 3.11 or 3.12. These steps are for Windows. On Mac or Linux, use `source venv/bin/activate` instead of `venv\Scripts\activate`.

git clone <https://github.com/jhenkar-bit/Handwritten-Digits-Recognition-system.git>
cd <Handwritten-Digits-Recognition-system>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


TensorFlow is a big download, so the install can take 10 to 15 minutes.

Running it

You need two terminal windows, both opened in the project folder.

Terminal 1 (the server):

```
venv\Scripts\activate
uvicorn backend:app --port 8000
```

Wait until it says `Application startup complete`.

Terminal 2 (the web page):

```
venv\Scripts\activate
streamlit run app.py
```

Your browser should open at `http://localhost:8501`. If it does not, type that address in yourself.

Keep both windows open while you use the app. Press Ctrl + C in each one to stop.

Tips for better results

- Draw the digit big and thick so it fills most of the box.
- For photos, use dark ink on light paper and take the picture from straight above, so the lines on the page run level.
- Leave a small gap between digits. If two digits touch, they are read as one shape.
- Write all the digits at roughly the same size.
- Try to get even light with no shadow across the page.

Limitations

- The model only learned from MNIST, so it works best on handwriting that looks similar. Some styles, for example a 9 with an open top that looks like a q, can be read wrong.
- Printed text and letters are not supported. It only knows the digits 0 to 9.
- Touching digits are not split apart.
- A very tilted photo can make the line removal fail.

Built with

- Python
- TensorFlow / Keras
- FastAPI and Uvicorn
- Streamlit and streamlit-drawable-canvas
- OpenCV, Pillow, NumPy, pandas

Notes

If you want to use a different model, save it as `digit_model.keras` in the project folder. The server checks the model's input shape when it starts, so it works with models that take 28x28 images or a flat list of 784 values.
