import sys, os, processor 

class BlackBox:
    def __init__(self):
        self.processor = processor.findFaceGetPulse(bpm_limits=[45, 130], data_spike_limit=250.,face_detector_smoothness=10.)
        self.processor.find_faces = False

    def loop(self, frame, forehead):
        self.processor.frame_in = frame
        self.processor.forehead = forehead
        return self.processor.run(0) #(bpm, alpha, text)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name='plot')
