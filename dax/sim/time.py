from operator import itemgetter

import vcd
import vcd.writer

import artiq.language.core
from artiq.language.units import *

import dax.util.units


class SequentialTimeContext:
    def __init__(self, current_time):
        self.current_time = current_time
        self.block_duration = 0.0

    def take_time(self, amount):
        self.current_time += amount
        self.block_duration += amount


class ParallelTimeContext(SequentialTimeContext):
    def take_time(self, amount):
        if amount > self.block_duration:
            self.block_duration = amount


class Manager:
    def __init__(self, file_name='out.vcd'):
        # TODO, name should preferably be configurable but with a default
        # TODO, timescale configurable?

        # Initialize stack
        self.stack = [SequentialTimeContext(0.0)]

        # Initialize timeline and timescale
        self._timeline = list()
        self._timescale = ns

        # Open file and instantiate VCD writer
        self._vcd_file = open(file_name, mode='w')
        self._vcd_writer = vcd.VCDWriter(self._vcd_file,
                                         timescale=dax.util.units.time_to_str(self._timescale, precision=0))

    def enter_sequential(self):
        new_context = SequentialTimeContext(self.get_time_mu())
        self.stack.append(new_context)

    def enter_parallel(self):
        new_context = ParallelTimeContext(self.get_time_mu())
        self.stack.append(new_context)

    def exit(self):
        old_context = self.stack.pop()
        self.take_time(old_context.block_duration)

    def take_time_mu(self, duration):
        self.stack[-1].take_time(duration)

    def get_time_mu(self):
        return self.stack[-1].current_time

    def set_time_mu(self, t):
        dt = t - self.get_time_mu()
        if dt < 0.0:
            # Going back in time is not allowed by the VCD writer
            raise ValueError("Attempted to go back in time")
        self.take_time_mu(dt)

    def take_time(self, duration):
        self.take_time_mu(duration // self._timescale)

    @artiq.language.core.rpc(flags={"async"})  # Added for accuracy, but does not imply an actual RPC call
    def event(self, var, value):
        self._timeline.append((self.get_time_mu(), var, value))

    def register(self, scope, name, var_type, size=None, init=None, ident=None):
        return self._vcd_writer.register_var(scope, name, var_type, size, init, ident)

    def format_timeline(self, ref_period):
        ref_timescale = ref_period // self._timescale  # TODO, ref_timescale actually does not change
        for time, var, value in sorted(self._timeline, key=itemgetter(0)):
            # Add events to the VCD file after time conversion
            self._vcd_writer.change(var, time * ref_timescale, value)

        # Flush the VCD file
        self._vcd_writer.flush()

        # Clear events from timeline
        self._timeline.clear()

    def __del__(self):
        # TODO, this function is maybe not actually called (and __delete__ neither)
        # Close the VCD file
        self._vcd_writer.close()
        self._vcd_file.close()


# TODO, implement the manager singleton in a more elegant way
# TODO, make a more elegant way to access the manager for registering signals (in a flavour matching with ARTIQ)

# Set the time manager
manager = Manager()
artiq.language.core.set_time_manager(manager)
