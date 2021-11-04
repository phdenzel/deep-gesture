

# deep<sub>gesture</sub>

A LSTM gesture recognition neural net which can be trained to
categorize any number of given gestures.

Key features:

-   easy to train
-   uses MediaPipe to generate a Holistic model
-   uses a custom TensorFlow LSTM neural net


## Requirements

-   `python3`
-   `pipenv` (for dev features)


## Install

Simply type `pip install deep_gesture`.

To install from source, you may type

    pipenv install --dev
    pipenv install -e .


## Usage

For more information type

    [pipenv run] deep_gesture -h

Run deep<sub>gesture</sub> in training mode (use webcam for data collection):

    [pipenv run] deep_gesture --train --device 0

Run deep<sub>gesture</sub> in streaming mode (use webcam for real-time gesture recognition):

    [pipenv run] deep_gesture --device 0

Run deep<sub>gesture</sub> in file mode (use video file to categorize a gesture):

    [pipenv run] deep_gesture --file example.mp4

Run deep<sub>gesture</sub> in test-mode:

    [pipenv run] deep_gesture -t

or with `pytest`:

    [pipenv run] pytest -v --cov=deep_gesture --cov-report=html

