"""
deep-gesture.camera

@author: phdenzel
"""
import numpy as np
from itertools import zip_longest
from functools import wraps
import cv2
from matplotlib import pyplot as plt


class CVFeed(object):

    feedname = 'OpenCV-Feed'

    def __init__(self, device=0):
        self.device = device
        self.init_capture()
        self.mod_sequence = {}
        self.mod_output = []

    def init_capture(self, device=None):
        """
        Set up capture device for read-out from opencv feed

        Kwargs:
            device <int|str> - device id, video filename, or IP video stream
        """
        if device is not None:
            self.device = device
        self.capture = cv2.VideoCapture(self.device)

    def close(self):
        """
        Stop capture device feed
        """
        if hasattr(self, 'display'):
            del self.display
        self.capture.release()
        cv2.destroyAllWindows()

    def init_mpl(self, figsize=(2, 2), dpi=200, hide_tk_toolbar=False):
        """
        Set up display framework for matplotlib backend

        Kwargs:
            figsize <float, float> - width and height in inches
            dpi <float> - resolution of the figure in dots-per-inch
            hide_tk_toolbar <bool> - remove Tk toolbar from figure display
        """
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')
        if hide_tk_toolbar:
            self.fig.canvas.toolbar.pack_forget()
        frame = self.capture.read()[1] if self.capture.isOpened() else np.zeros((1, 1))
        self.display = self.ax.imshow(frame, aspect='equal')
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.key = self.fig.number

    def init_nb(self):
        """
        Set up display framework for IPython kernel

        Note:
            async issue not solved yet! 'mpl' seems to work on notebooks too...
        """
        from IPython.display import display, Image
        import ipywidgets as widgets
        # display setup
        self.display = display(None, display_id=True)
        def jupyter_display_stream(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, image = cv2.imencode('.jpg', image)
            image = Image(data=image.tobytes())
            self.display.update(image)
        self.display.set_data = jupyter_display_stream
        # cancel button
        # TODO: fix async problem
        self.key = False
        self.button = widgets.ToggleButton(description='Cancel')
        self.button.observe(self.key_change_nb, names='value')
        display(self.button)

    def key_change_nb(self, widget):
        """
        Button-press event trigger function
        """
        self.key = not self.key
        print(self.key)

    @staticmethod
    def standardize_feed(frame):
        """
        A set of operations on a feed frame (just) after read-out,
        mainly converts from BGR to RGB

        Args:
            frame <np.ndarray> - BGR image

        Return:
            frame <np.ndarray> - RGB image
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def single_capture(self, actions=True, ignore_signal=True):
        """
        Single-image feed read-out

        Kwargs:
            actions <bool> - use mod_sequence for feed image manipulation
            ignore_signal <bool> - do not return feed read-out status signal

        Return:
            [signal <int> - feed read-out status code]
            frame <np.ndarray> - RGB image
        """
        signal, frame = self.capture.read()
        frame = self.standardize_feed(frame)
        self.image = frame
        # image manipulations
        if actions:
            for func in self.mod_sequence:
                args = self.mod_args(func)
                kwargs = self.mod_kwargs(func)
                out = func(*args, **kwargs)
                self.mod_output.append(out)
        self.close()
        self.init_capture()
        if ignore_signal:
            return frame
        else:
            return signal, frame

    def start(self, display_handler='mpl', actions=True):
        """
        Start opencv feed loop and display

        Kwargs:
            display_handler <str> - display handler mode selection
                                    ('mpl' | 'nb' | 'cv' | None)
            actions <bool> - use mod_sequence for feed image manipulation

        Note: 
            Using cv2.imshow for display would be easier, 
            but throws segfaults on several systems!
        """
        is_display = hasattr(self, 'display')
        if not is_display and display_handler == 'mpl':
            self.init_mpl()
        elif not is_display and display_handler == 'nb':
            self.init_mpl(hide_tk_toolbar=False)
        elif not is_display:
            self.init_mpl()
        while self.capture.isOpened():
            signal, frame = self.capture.read()
            self.image = self.standardize_feed(frame)
            # image manipulations
            self.mod_output = []
            if actions:
                for func in self.mod_sequence:
                    args = self.mod_args(func)
                    kwargs = self.mod_kwargs(func)
                    out = func(*args, **kwargs)
                    self.mod_output.append(out)
            # display image from feed
            self.display_update(display_handler)
            # break gracefully
            if self.__getattribute__("break_{}".format(display_handler))():
                plt.close()
                break
        self.close()

    def display_update(self, display_handler='mpl'):
        """
        Feed update on display

        Kwargs:
            display_handler <str> - display handler mode selection
                                    ('mpl' | 'nb' | 'cv' | None)
        """
        image = cv2.flip(self.image, 1)
        if display_handler == 'mpl':   # matplotlib
            self.display.set_data(image)
            plt.pause(0.01)
        elif display_handler == 'nb':  # jupyter-notebook
            self.display.set_data(image)
            self.fig.canvas.draw()
            plt.pause(0.05)
        elif display_handler == 'cv':  # opencv
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(image)
            self.key = cv2.waitKey(10)

    def register_mod(self, func,
                     feed_passthru=False, args=(),
                     output_passthru={}, kwargs={}):
        """
        Add a feed manipulation specification to mod_sequence

        Args:
            func <callable> - next mod function in sequence

        Kwargs:
            feed_passthru <bool> - include last feed image to mod function args
            args <tuple> - additional mod function args
            output_passthru <dict> - include any outputs from previous
                                     mod function calls;
                                     format: {output index <int[, int]>: key: <str>}
            kwargs <dict> - additional mod function kwargs

        The mod function call is as follows:
        func(*args, **kwargs) 
            where feed_passthru & args -> *args
                  output_passthru & kwargs -> **kwargs
        """
        self.mod_sequence[func] = {'feed_passthru': feed_passthru,
                                   'output_passthru': output_passthru,
                                   'args': args,
                                   'kwargs': kwargs}

    def mod_args(self, func):
        """
        Extract the corresponding function arguments of the current instance
        state from the mod_sequence

        Args:
            func <callable> - mod function in sequence
        """
        mod_specs = self.mod_sequence[func]
        args = (self.image,) if mod_specs['feed_passthru'] else ()
        args = args + mod_specs['args']
        return args

    def mod_kwargs(self, func):
        """
        Extract the corresponding function keyword arguments of the current
        instance state from the mod_sequence

        Args:
            func <callable> - mod function in sequence
        """
        mod_specs = self.mod_sequence[func]
        kwargs = mod_specs['kwargs']
        out_passthru = {}
        for output_idx in mod_specs['output_passthru']:
            pass  # TODO
        kwargs = {**out_passthru, **kwargs}
        return kwargs

    def break_mpl(self):
        """
        Breaking condition for 'mpl' display handler
        """
        return not plt.fignum_exists(self.key)

    def break_nb(self):
        """
        Breaking condition for 'nb' display handler
        """
        return not plt.fignum_exists(self.key)

    def break_cv(self):
        """
        Breaking condition for 'cv' display handler
        """
        return self.key & 0xFF == 27

    def start_count(self, counts=[], break_condition=None):
        pass


if __name__ == "__main__":
    from holistic import HolisticMP
    holistic = HolisticMP()
    cvstream = CVFeed(device=0)
    cvstream.register_mod(holistic.detection, feed_passthru=True)
    cvstream.register_mod(holistic.draw, feed_passthru=True, args=('face', 'pose', 'right_hand'))
    cvstream.start()
    cvstream.close()
    
