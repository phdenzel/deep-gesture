"""
deep_gesture.process

@author: phdenzel
"""
import numpy as np
import deep_gesture as dg


class DeepGestureProcessor(object):
    """
    High-level data and model processor
    """
    def __init__(self, holistic, configurator, sequence_length=None,
                 cache_size=10):
        self.holistic = holistic
        self.configurator = configurator
        self.inference_map = self.configurator.inv_label_map
        self.inference_map[None] = None
        self.model_loaded = hasattr(self.configurator, 'model')
        if sequence_length is None:
            sequence_length = dg.sequence_length
        self.sequence = np.zeros((sequence_length,)+self.holistic.landmarks.shape)
        self.inference = -1
        self.classification = None
        self.stack = [None] * max(cache_size, 1)

    def update_sequence(self, *args):
        self.sequence[0:-1] = self.sequence[1:]
        self.sequence[-1] = self.holistic.landmarks

    def classify(self, *args, threshold=0.95, verbose=False):
        sequence_complete = np.all(np.sum(self.sequence, axis=1))
        if sequence_complete and self.model_loaded:
            self.inference = self.configurator.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            # if self.max_prob >= threshold:
            self.stack[0:-1] = self.stack[1:]
            self.stack[-1] = np.argmax(self.inference)
            if np.all(self.stack) and np.min(self.stack) == np.max(self.stack):
                self.classification = self.inference_map[self.stack[-1]]

    @property
    def max_prob(self):
        return np.max(self.inference)

    def show_probs(self, cvstream):
        if np.sum(self.inference) < 0:
            return
        label_set = list(self.configurator.label_map.keys())
        colors = cvstream.generate_colors(len(label_set))
        y_range = 0.0, 0.15
        x_range = 0.1, 0.15
        x_inc = 0.0625
        for label in label_set:
            idx = self.configurator.label_map[label]
            start = y_range[0], x_range[0]+idx*x_inc
            end = y_range[1]*self.inference[idx], x_range[1]+idx*x_inc
            cvstream.bar_to_image(cvstream.image, start, end, colors[idx])
            cvstream.text_to_image(cvstream.image, text=label, position=(end[1]-0.01, 0))


def proc_feed():
    """
    Starts processing procedure and analyzes feed
    """
    holistic = dg.holistic.HolisticMP()
    cvstream = dg.camera.CVFeed(device=dg.device_id, display_handler='cv')
    configurator = dg.models.ModelConfigurator.from_archive(
        # model_name='284SUBVCVVVRI_Adam',
        from_checkpoints=True, verbose=True)
    processor = DeepGestureProcessor(holistic, configurator)
    

    cvstream.register_mod(processor.holistic.detection, feed_passthru=True)
    cvstream.register_mod(processor.holistic.draw, feed_passthru=True,
                          kwargs={'parts': ('face', 'pose', 'lh', 'rh')})
    cvstream.register_mod(cvstream.flip_feed, feed_passthru=True)
    cvstream.register_mod(processor.update_sequence, feed_passthru=True)
    cvstream.register_mod(processor.classify, feed_passthru=False,
                          kwargs={'threshold': 0.7})
    cvstream.register_mod(processor.show_probs, feed_passthru=False,
                          args=(cvstream,))
    cvstream.register_mod(cvstream.key_action, feed_passthru=True,
                          kwargs={'key_action_map': {'q': cvstream.close,
                                                     's': cvstream.write_frame}})

    cvstream.start()
    cvstream.close()


def proc_file():
    pass

