# ****************************************************************************
#
# Copyright (C) 2019-2020, ShakeLab Developers.
# This file is part of ShakeLab.
#
# ShakeLab is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# ShakeLab is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# with this download. If not, see <http://www.gnu.org/licenses/>
#
# ****************************************************************************
"""
Module for basic waveform analysis
"""

from scipy import signal
import numpy as np

from shakelab.signals import mseed, sac
from shakelab.libutils.time import Date


def import_record(file, format='sac', path=None, byte_order='le',
                     **kwargs):
    """
    """

    # Initialise an empty trace
    rec_list = []

    # Import recordings from file
    if format is 'mseed':
        ms = mseed.MiniSeed(file, byte_order=byte_order)
        for mr in ms.record:
            rec = Record()
            rec.dt = mr.sampling_rate()
            rec.time = mr.time_date()
            rec.data = np.array(mr.data)
            rec_list.append(rec)

    elif format is 'sac':
        sac = sac.Sac(file, byte_order=byte_order)
        rec = Record()
        rec.dt = sac.sampling_rate()
        rec.time = sac.time_date()
        rec.data = np.array(sac.data[0])

    elif format is 'ascii':
        raise NotImplementedError('format not yet implemented')

    elif format is 'seisan':
        raise NotImplementedError('format not yet implemented')

    elif format is 'seg2':
        raise NotImplementedError('format not yet implemented')

    elif format is 'dat':
        raise NotImplementedError('format not yet implemented')

    elif format is 'gse':
        raise NotImplementedError('format not yet implemented')

    else:
        raise NotImplementedError('format not recognized')

    return rec_list



class Record(object):
    """
    Single recording.
    """

    def __init__(self):
        self.dt = None
        self.data = []
        self.time = Date()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, sliced):
        return self.data[sliced]

    def rmean(self):
        """
        """
        self.data -= np.mean(self.data)

    def filter(self, low=None, high=None, order=2):
        """
        """

        # Corner frequencies
        corners = []

        if low:
            corners.append(2. * low * self.dt)
            filter_type = 'high'

        if high:
            corners.append(2. * high * self.dt)
            filter_type = 'low'

        if low and high:
            filter_type = 'band'

        if len(corners) > 0:
            # Butterworth filter
            sos = signal.butter(order, corners, analog=False,
                                btype=filter_type, output='sos')
            self.data = signal.sosfiltfilt(sos, self.data)

    def taper(self, window=0.1):
        """
        time is in seconds.
        negative time means the whole window (cosine taper)
        """
        tnum = len(self.data)
        if window < 0:
            alpha = 1
        else:
            alpha = 2 * float(window)/(self.dt * tnum)
        self.data = (self.data * signal.tukey(tnum, alpha))

    def cut(self, start, stop):
        """
        """
        pass

    def pad(self, zeros):
        """
        """
        pass

    def shift(self, time):
        """
        """
        pass