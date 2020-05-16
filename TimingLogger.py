from time import time

class TimingLogger(object):
    def __init__(self, fn, text):
        self.fn = fn
        self.text = text

    def __enter__(self):
        self.tstart = time()

    def __exit__(self, type, value, traceback):
        f = open(self.fn,'a')
        f.write('[%s]: %s\n' % (self.text, time() - self.tstart))
        f.close()