"""
Power Monitor Utilities

A collection of helper functions to control Monsoon Power Monitor using PyMonsoon.
"""

from __future__ import print_function


# This script currently assumes you're using the black High-Voltage Power
# Monitor. Un/comment lines below to use the older white Power Monitor.
USING_WHITE_POWER_MONITOR=False #TODO: Find a way to detect this at run-time
if USING_WHITE_POWER_MONITOR:
    import Monsoon.LVPM as PM
else:
    import Monsoon.HVPM as PM

import Monsoon.sampleEngine as sampleEngine
import Monsoon.reflash as reflash
from Monsoon import Operations as ops

from datetime import datetime
import os
import socket
import sys
import pickle

import logging
import coloredlogs
logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s %(levelname)s %(message)s'
coloredlogs.install(level='DEBUG', logger=logger, fmt=log_fmt) # Fancy colourful logs
LOGS_DIR='_power_measurement_logs'

def power_up_pi():
    """
    Sets the correct voltage on the main channel for Raspberry Pi"
    """

    # Set main output voltage in 0.01V increments. Valid values are:
    # LVPM: 2.01-4.55
    # HVPM: 0.8-13.5

    Mon = PM.Monsoon()
    Mon.setup_usb()

    if USING_WHITE_POWER_MONITOR:
        logger.warning("White Power Monsoon's maximum voltage is 4.2V. Raspberry Pi \
                        kind of works with this voltage but some parts (e.g. LEDs) may \
                        not work")

        Mon.setVout(4.2)
    else:
        Mon.setVout(5.0)

def turn_voltage_off():
    Mon = PM.Monsoon()
    Mon.setup_usb()
    Mon.setVout(0)

def upgrade_firmware():
    """
    Reflash white Power Monitor with the new USB Protocol firmware.
    """
    if not USING_WHITE_POWER_MONITOR:
        logger.error("upgrade_firmware() is only relevant for white Power Monitor")
        return;

    Mon = reflash.bootloaderMonsoon()
    Mon.setup_usb()
    Header, Hex = Mon.getHeaderFromFWM('./firmware_images/LVPM_RevE_Prot_1_Ver25_beta.fwm')
    if(Mon.verifyHeader(Header)):
        Mon.writeFlash(Hex)

def downgrade_firmmware():
    """
    Return white Power Monitor firmware to the original serial protocol firmware.
    """
    if not USING_WHITE_POWER_MONITOR:
        logger.error("downgrade_firmmware() is only relevant for white Power Monitor")
        return;

    Mon = reflash.bootloaderMonsoon()
    Mon.setup_usb()
    Hex = Mon.getHexFile('./firmware_images/PM_RevD_Prot17_Ver20.hex')
    Mon.writeFlash(Hex)

def live_main():
    """
    Not particularly useful but good for testing.
    """
    Mon = PM.Monsoon()
    Mon.setup_usb()

    engine = sampleEngine.SampleEngine(Mon)
    engine.ConsoleOutput(True)
    numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE

    # We shouldn't need to set up triggers with the numSamples defined above but
    # for some reason that does stop sampling after 50000 samples.
    engine.setTriggerChannel(sampleEngine.channels.MainCurrent)
    engine.setStopTrigger(sampleEngine.triggers.GREATER_THAN,9999)

    engine.startSampling(numSamples)
    Mon.closeDevice();

def live_usb():
    Mon = PM.Monsoon()
    Mon.setup_usb()

    engine = sampleEngine.SampleEngine(Mon)
    engine.ConsoleOutput(True)
    numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE

    engine.enableChannel(sampleEngine.channels.USBCurrent)
    engine.enableChannel(sampleEngine.channels.USBVoltage)
    engine.disableChannel(sampleEngine.channels.MainCurrent)
    engine.disableChannel(sampleEngine.channels.MainVoltage)
    Mon.setUSBPassthroughMode(ops.USB_Passthrough.Off)

    engine.startSampling(numSamples)
    Mon.closeDevice();

def trigger_and_capture(get_test_details_remotely=False, remote_ip=None):
    """
    Triggers on the voltage on the USB channel and captures samples from the main
    channel into a log file in the specified log directory
    """
    Mon = PM.Monsoon()
    Mon.setup_usb()

    engine = sampleEngine.SampleEngine(Mon)

    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    log_file = _get_timestamped_filepath()
    engine.enableCSVOutput(log_file)
    engine.ConsoleOutput(False)

    numSamples = sampleEngine.triggers.SAMPLECOUNT_INFINITE

    engine.enableChannel(sampleEngine.channels.USBVoltage)
    engine.setTriggerChannel(sampleEngine.channels.USBVoltage)
    engine.setStartTrigger(sampleEngine.triggers.GREATER_THAN,3.0)
    engine.setStopTrigger(sampleEngine.triggers.LESS_THAN,1.0)

    try:
        logger.info('Trigerring on USB voltage ...')
        engine.startSampling(numSamples)
        logger.info('Capture completed: %s', log_file)

        if get_test_details_remotely:
            test_name = _request_remote_test_details(remote_ip)
    finally:
        #TODO: If log file is empty delete it?
        Mon.closeDevice();

def _request_remote_test_details(remote_ip):
    """
    TODO
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        logger.info('Socket Created')

        s.connect((remote_ip , 8887))
        logger.info('Socket Connected to ' + remote_ip + ':8887')

        # Pickle uses different default protocols in Python 2 and 3 so I'm
        # specifying a common one explicitly.
        serialised_message = pickle.dumps("A random message", protocol=0)

        s.sendall(serialised_message)
        logger.info('Request sent successfully')

        # Blocking call
        serialised_reply = s.recv(4096)

        logger.info('Raw reply {}'.format(serialised_reply))

        reply = pickle.loads(serialised_reply)

        logger.info('Received test details: {}'.format(reply))

        return reply
    # except:
    #     logger.error('Something wrong with socket connection')
    #     sys.exit()
    finally:
        s.close()

def _get_timestamped_filepath():
    timestamp = datetime.now().strftime('%d-%b-%Y-%X')
    return LOGS_DIR + '/' + timestamp + '_pm_capture.csv'
