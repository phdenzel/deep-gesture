"""
deep_gesture.record

@author: phdenzel
"""
import os
import cv2
import deep_gesture as dg

__all__ = ['collect_training_data']


def training_sequences(gestures=[], sequences=1, length=None):
    """
    Get default values for standardized input for indices 
    from deep_gesture.camera.CVFeed.start_iter
q
    Kwargs:
        gestures <list(str)> - gesture names as data labels
        sequences <int> - number of sequences for each gesture
        length <int> - number of frames for each gesture sequence

    Return:
        iteration indices <list(iterable)>
    """
    if not gestures:
        gestures = dg.gestures
    if sequences is None:
        sequences = dg.N_training_sequences
    if length is None:
        length = dg.sequence_length
    return [gestures, range(sequences), range(length)]


def sequence_str(*args):
    """
    Sequence string to be displayed on the video feed

    Args:
        args <*tuple> - some name, name ID, and frame number
    """
    msg = "{} - Sequence {} - Frame {}".format(*args)
    return msg


def collect_training_data():
    """
    Starts training procedure and records data
    """
    holistic = dg.holistic.HolisticMP()
    cvstream = dg.camera.CVFeed(device=dg.device_id, display_handler='cv')
    gestures = dg.gestures
    N_sequences = dg.N_training_sequences
    length = dg.sequence_length

    cvstream.register_mod(holistic.detection, feed_passthru=True)
    cvstream.register_mod(holistic.draw, feed_passthru=True,
                          kwargs={'parts': ('face', 'pose', 'lh', 'rh')})
    cvstream.register_mod(cvstream.flip_feed, feed_passthru=True)
    cvstream.register_mod(cvstream.text_to_image, feed_passthru=True,
                          kwargs={'text': sequence_str, 'position': (0.1, 0.1)})
    cvstream.register_mod(cvstream.pause, feed_passthru=False,
                          kwargs={'pause': 3})
    cvstream.register_mod(holistic.save_landmarks, feed_passthru=True,
                          kwargs={'save_dir': dg.TMP_DIR})
    cvstream.register_mod(cvstream.write_frame, feed_passthru=True)
    cvstream.register_mod(cvstream.await_response, feed_passthru=False)
    cvstream.register_mod(cvstream.key_action, feed_passthru=True,
                          kwargs={'key_action_map': {'q': cvstream.close,
                                                     'd': dg.utils.clean_dir,
                                                     '*': dg.utils.archive_data}})

    cvstream.start_iter(indices=training_sequences(gestures, N_sequences, length))
    cvstream.close()
