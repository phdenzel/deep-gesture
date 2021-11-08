"""
deep_gesture.camera

@author: phdenzel
"""
import os
from itertools import product as iter_prod
import numpy as np
import cv2
from matplotlib import pyplot as plt
import deep_gesture as dg
# from functools import wraps


class CVFeed(object):

    feedname = 'OpenCV-Feed'

    def __init__(self, device=0, display_handler='cv'):
        """
        Kwargs:
            device <int|str> - device id, video filename, or IP video stream
            display_handler <str> - display handler mode selection
                                    ('cv' | 'mpl' | 'nb' | None)
        """
        # camera settings
        self.device = device
        self.display_handler = display_handler
        self.init_capture()
        # a set of instructions to run within feed loop
        self.mod_sequence = {}
        self.mod_output = []
        # keyboard response
        self.key = 0

    def init_capture(self, device=None):
        """
        Set up capture device for read-out from opencv feed

        Kwargs:
            device <int|str> - device id, video filename, or IP video stream
        """
        if device is not None:
            self.device = device
        self.capture = cv2.VideoCapture(self.device)
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)

    def close(self):
        """
        Stop capture device feed
        """
        plt.close()
        if hasattr(self, 'display'):
            del self.display
        self.capture.release()
        cv2.destroyAllWindows()
        # TODO: clean TMP_DIR

    def init_display(self, display_handler=None):
        """
        Set up display framework depending on handler choice

        Kwargs:
            display_handler <str> - display handler mode selection
                                    ('cv' | 'mpl' | 'nb' | None)
        """
        if display_handler is None:
            display_handler = self.display_handler
        is_display = hasattr(self, 'display')
        if not is_display and display_handler == 'mpl':
            self.init_mpl()
        elif not is_display and display_handler == 'nb':
            # self.init_nb()  # TODO: once init_nb is functional
            self.init_mpl(hide_tk_toolbar=False)

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

        Args:
            TODO
        """
        self.key = not self.key
        print(self.key)

    def register_mod(self, func,
                     feed_passthru=False, args=(),
                     output_passthru={}, kwargs={}):
        """
        Add a feed manipulation specification to mod_sequence

        Args:
            func <callable> - next mod function in sequence

        Kwargs:
            feed_passthru <bool> - include last feed image and any feed 
                                   iteration info to mod function args
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
        if mod_specs['feed_passthru']:
            args = (self.image,)
            args = args+self.iter_specs if hasattr(self, 'iter_specs') else args
        else:
            args = ()
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

    @staticmethod
    def standardize_feed(frame, *args):
        """
        A set of operations on a feed frame (just) after read-out,
        mainly converts from BGR to RGB

        Args:
            frame <np.ndarray> - BGR image
            args <*tuple> - dummy arguments for compatibility

        Return:
            frame <np.ndarray> - RGB image
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def flip_feed(self, *args):
        """
        Flip feed frame

        Args:
            args <*tuple> - dummy arguments for compatibility
        """
        self.image = cv2.flip(self.image, 1)

    @staticmethod
    def text_to_image(frame, *args, text="", position=(0.5, 0.5),
                      fontsize=1, color=(255, 255, 255), thickness=4):
        """
        Write text on the feed image

        Args:
            frame <np.ndarray> - RGB image
            args <*tuple> - if text is callable, args can optionally be passed 
                            as input

        Kwargs:
            text <str|callable> - text to write on image
            position <tuple> - relative position on the image
            fontsize <int> - fontsize of the text
            color <tuple(int)> - RGB values between 0-255
            thickness <int> - thickness of the text
        """
        if callable(text):
            text = text(*args)
        x, y, _ = frame.shape
        position = (int(x*position[0]), int(y*position[1]))
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    fontsize, color, thickness, cv2.LINE_AA)

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
        signal = self.grab_feed()
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

    def grab_feed(self):
        """
        Grab a standardized RGB feed image
        """
        signal, frame = self.capture.read()
        if signal:
            self.image = self.standardize_feed(frame)
        return signal

    def start(self, display_handler=None, actions=True):
        """
        Start opencv feed loop and display

        Kwargs:
            display_handler <str> - display handler mode selection
                                    ('cv' | 'mpl' | 'nb' | None)
            actions <bool> - use mod_sequence for feed image manipulation

        Note: 
            Using cv2.imshow for display would be best,
            but throws segfaults for Python 3.9!
        """
        if display_handler is None:
            display_handler = self.display_handler
        self.init_display(display_handler=display_handler)
        # start feed loop
        while self.capture.isOpened():
            signal = self.grab_feed()
            if not signal:
                continue
            # image manipulations
            self.mod_output = []
            if actions:
                for func in self.mod_sequence:
                    args = self.mod_args(func)
                    kwargs = self.mod_kwargs(func)
                    out = func(*args, **kwargs)
                    self.mod_output.append(out)
                    # # Skip out of mod sequence and continue
                    # if self.__getattribute__("skip_{}".format(display_handler))():
                    #     break
            # display image from feed
            self.update_display(display_handler)
            # break gracefully
            if self.__getattribute__("break_{}".format(display_handler))():
                break
        self.close()

    def start_iter(self, indices=[], display_handler=None, actions=True):
        """
        Start opencv feed loop and display

        Kwargs:
            indices <list(iterables)> - sequences with which to time the feed
            display_handler <str> - display handler mode selection
                                    ('cv' | 'mpl' | 'nb' | None)
            actions <bool> - use mod_sequence for feed image manipulation
        """
        if display_handler is None:
            display_handler = self.display_handler
        self.init_display(display_handler=display_handler)
        # start feed loop
        self.iter_last = indices[-1][-1]
        for iter_specs in iter_prod(*indices):
            self.iter_specs = iter_specs
            signal = self.grab_feed()
            if not signal:
                continue
            for func in self.mod_sequence:
                    args = self.mod_args(func)
                    kwargs = self.mod_kwargs(func)
                    out = func(*args, **kwargs)
                    self.mod_output.append(out)
                    # # Skip out of mod sequence and continue
                    # if self.__getattribute__("skip_{}".format(display_handler))():
                    #     break
            # display image from feed
            self.update_display(display_handler)
            # break gracefully
            if self.__getattribute__("break_{}".format(display_handler))():
                break
        self.close()

    def update_display(self, display_handler=None):
        """
        Feed update on display

        Kwargs:
            display_handler <str> - display handler mode selection
                                    ('cv' | 'mpl' | 'nb' | None)
        """
        image = self.image
        if image is None:
            return
        if display_handler is None:
            display_handler = self.display_handler
        if display_handler == 'cv':  # opencv
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.feedname, image)
            self.key = cv2.waitKey(10) & 0xFF
        elif display_handler == 'mpl':  # matplotlib
            self.display.set_data(image)
            plt.pause(0.01)
        elif display_handler == 'nb':  # jupyter-notebook
            self.display.set_data(image)
            self.fig.canvas.draw()
            plt.pause(0.05)

    def write_frame(self, *args, filename=None, save_dir=None, verbose=True):
        """
        Write current feed frame to disk

        Args:
            args <*tuple> - optional positional arguments;
                            (image, prefix, name_id, part_no, etc.)
                            see dg.utils.generate_filename

        Kwargs:
            filename <str> - name of the file
            save_dir <str> - path to directory in which to save the image
            verbose <bool> - print information to stdout
        """
        if args:
            image, args = args[0], args[1:]
        else:
            image, args = self.image, ()
        fname = dg.utils.generate_filename(*args, extension='jpg') \
            if filename is None else filename
        save_dir = dg.TMP_DIR if save_dir is None else save_dir
        dg.utils.mkdir_p(save_dir)
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            print('Image filename already exists... rerolling filename!')
            self.write_frame(image, *args,
                             filename=filename, save_dir=save_dir,
                             verbose=verbose)
        if verbose:
            print(fpath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(fpath, image)

    def pause(self, *args, pause=5, pause_step=1, display_handler=None, static=False):
        """
        Pause within an iterated loop
        
        Args:
            args <*tuple> - dummy arguments for compatibility; 

        Kwargs:
            pause <int> - pause in approx. seconds
            pause_step <int> - step between pauses to update in approx. seconds
            display_handler <str> - display handler mode selection
                                    ('cv' | 'mpl' | 'nb' | None)
            static <bool> - pause display feed during pause

        Note:
            only display_handler='cv' implemented for now
        """
        pause *= 1000
        pause_step *= 1000
        idx_frame = 0
        if display_handler is None:
            display_handler = self.display_handler
        if hasattr(self, 'iter_specs'):
            idx_frame = self.iter_specs[-1]
        if idx_frame == 0:
            while pause > 0:
                signal = self.grab_feed()
                if not signal:
                    continue
                self.flip_feed()
                self.text_to_image(
                    self.image,
                    text="Starting in {}".format(int(pause//pause_step)+1),
                    position=(0.35, 0.5))
                if not static:
                    key_cache = self.key*1
                    self.update_display()
                    self.key = key_cache
                pause = pause - int(2*self.fps)
                self.key = cv2.waitKey(int(self.fps)) & 0xFF
                # break gracefully
                if self.__getattribute__("break_{}".format(display_handler))():
                    self.close()
                    break

    def await_response(self, *args):
        """
        Wait for key press and display instructions;
        if feed is iterated this is only executes on the last frame

        Args:
            args <*tuple> - dummy arguments for compatibility; 
        """
        idx_frame = last_frame = 0
        if hasattr(self, 'iter_specs'):
            idx_frame = self.iter_specs[-1]
            last_frame = self.iter_last
        if idx_frame == last_frame:
            signal = self.grab_feed()
            self.flip_feed()        
            self.text_to_image(self.image, text="Type 'd' to skip",
                               position=(0.25, 0.3))
            self.text_to_image(self.image, text="Type 'q' to quit",
                               position=(0.25, 0.4))
            self.text_to_image(self.image, text="Type any other key to save",
                               position=(0.25, 0.5))
            key_cache = self.key*1
            self.update_display()
            self.key = key_cache
            self.wait_for_key()

    def wait_for_key(self, *args):
        """
        Stop feed until a key is pressed

        Args:
            args <*tuple> - dummy arguments for compatibility; 

        Note:
            only display_handler='cv' implemented for now
        """
        self.key = cv2.waitKey(-1) & 0xFF

    def key_action(self, *args, key_action_map={}):
        """
        Defines a set of instructions executed on given key presses;
        if feed is iterated this is only executes on the last frame

        Args:
            args <*tuple> - dummy arguments for compatibility; 

        Kwargs:
            key_action_map <dict(str/int=callable)> - 
                keys are keys or key ordinals, values are callables; 
                e.g. {'q': self.close}
        """
        idx_frame = last_frame = 0
        if hasattr(self, 'iter_specs'):
            idx_frame = self.iter_specs[-1]
            last_frame = self.iter_last
        if idx_frame == last_frame:
            for key in key_action_map:
                if ord(key) == self.key or key == self.key:
                    key_action_map[key]()
                    break
                elif key == '*':
                    key_action_map['*']()
                    break

    def break_cv(self):
        """
        Breaking condition for 'cv' display handler
        """
        return self.key == ord('q')

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


if __name__ == "__main__":
    from deep_gesture.holistic import HolisticMP
    holistic = HolisticMP()
    cvstream = CVFeed(device=0, display_handler='cv')
    cvstream.register_mod(holistic.detection, feed_passthru=True)
    cvstream.register_mod(holistic.draw, feed_passthru=True,
                          parts=('face', 'pose', 'right_hand'))
    cvstream.register_mod(cvstream.flip_feed, feed_passthru=False)
    cvstream.register_mod(cvstream.update_display, feed_passthru=False)
    # cvstream.register_mod(cvstream.pause, feed_passthru=False,
    #                       kwargs=dict(pause=5, static=False))
    cvstream.start()
    cvstream.close()
    
