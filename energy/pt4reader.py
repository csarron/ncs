#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# This file is part of pt4utils.
#
# Copyright (C) 2013, Marcelo Martins.
#
# pt4utils was developed in affiliation with Brown University,
# Department of Computer Science. For more information about the
# department, see <http://www.cs.brown.edu/>.
#
# pt4utils is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pt4utils is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pt4utils.  If not, see <http://www.gnu.org/licenses/>.
#

import sys
import datetime
import os
import struct


class Header(object):
    def __init__(self, headerSize, identifier, batterySize, captureDate, serial,
                 calibrationStatus, voutSetting, voutValue, hardwareRate,
                 softwareRate, powerField, currentField, voltageField,
                 applicationInfo, samples, sum_):
        self.headerSize = headerSize
        self.identifier = identifier
        self.batterySize = batterySize
        self.captureData = captureDate
        self.serial = serial
        self.softwareRate = softwareRate
        self.powerField = powerField
        self.currentField = currentField
        self.voltageField = voltageField
        self.applicationInfo = applicationInfo
        self.samples = samples
        self.sum_ = sum_


class ApplicationInfo(object):
    def __init__(self, captureSetting, swVersion, runMode, exitCode):
        self.captureSetting = captureSetting
        self.swVersion = swVersion
        self.runMode = runMode
        self.exitCode = exitCode


class Samples(object):
    def __init__(self, captureDataMask, totalCount, statusOffset, statusSize,
                 sampleOffset, sampleSize):
        self.captureDataMask = captureDataMask
        self.totalCount = totalCount
        self.statusOffset = statusOffset
        self.statusSize = statusSize
        self.sampleOffset = sampleOffset
        self.sampleSize = sampleSize


class SumValues(object):
    def __init__(self, voltage, current, power):
        self.voltage = voltage
        self.current = current
        self.power = power


class Sum(object):
    def __init__(self, main, usb, aux):
        self.main = main
        self.usb = usb
        self.aux = aux


class Constants(object):
    missingRawCurrent = 0x7001
    missingRawVoltage = 0xFFFF
    coarseMask = 1
    marker0Mask = 1
    marker1Mask = 2
    markerMask = (marker0Mask | marker1Mask)


class RawSample(object):
    """ Wrapper for raw sample collected from Monson power monitor """

    def __init__(self, mainCurrent, usbCurrent, auxCurrent, voltage):
        self.mainCurrent = mainCurrent
        self.usbCurrent = usbCurrent
        self.auxCurrent = auxCurrent
        self.voltage = voltage

        # Voltage missing from recorded data?
        self.voltageMissing = Voltage(voltage).missing
        # Main current missing from recorded data?
        self.mainCurrentMissing = Current(mainCurrent).missing
        # USB current missing from recorded data?
        self.usbCurrentMissing = Current(usbCurrent).missing
        # Aux current missing from recorded data?
        self.auxCurrentMissing = Current(auxCurrent).missing
        # Any sample fields missing from recorded data?
        self.missing = (self.voltageMissing |
                        self.mainCurrentMissing |
                        self.usbCurrentMissing |
                        self.auxCurrentMissing)


class Voltage(object):
    """ Data Converter (raw sample -> voltage) """

    def __init__(self, raw):
        self.raw = raw
        # Voltage is missing from recorded data?
        self.missing = (self.raw == Constants.missingRawVoltage)
        # Recorded data has marker 0 channel?
        self.hasMarker0 = (self.raw & Constants.marker0Mask) != 0
        # Recorded data has marker 1 channel?
        self.hasMarker1 = (self.raw & Constants.marker1Mask) != 0

    def toVolts(self):
        """ Convert voltage to V """
        return (self.raw & (~Constants.markerMask)) * 125.0 / 1e6

    def toMarker(self):
        marker = 'none'
        if self.hasMarker0 and self.hasMarker1:
            marker = 'both'
        if self.hasMarker0:
            marker = 'marker0'
        if self.hasMarker1:
            marker = 'marker1'
        return marker


class Current(object):
    """ Data converter (raw sample -> current) """

    def __init__(self, raw):
        self.raw = raw
        # Current is missing from recorded data?
        self.missing = (self.raw == Constants.missingRawCurrent)

    def toMilliAmps(self):
        """ Convert current to mA """
        isCoarse = (self.raw & Constants.coarseMask) != 0
        mA = (self.raw & (~Constants.coarseMask)) / 1000.0
        return (mA * 250.0) if isCoarse else mA


class Sample(object):
    """ Wrapper for sample from Monsoon power monitor """

    def __init__(self, mainCurrent, usbCurrent, auxCurrent, voltage, marker):
        self.mainCurrent = mainCurrent
        self.usbCurrent = usbCurrent
        self.auxCurrent = auxCurrent
        self.voltage = voltage
        self.marker = marker

    def __str__(self):
        return "{:4.1f},{:8.4f},{}".format(self.mainCurrent, self.mainCurrent * self.voltage, self.marker)

    @staticmethod
    def fromRaw(rawSample, statusPacket):
        """ Create and return sample from raw data """
        mainCurrent = Current(rawSample.mainCurrent)
        usbCurrent = Current(rawSample.usbCurrent)
        auxCurrent = Current(rawSample.auxCurrent)
        voltage = Voltage(rawSample.voltage)

        return Sample(mainCurrent.toMilliAmps(), usbCurrent.toMilliAmps(),
                      auxCurrent.toMilliAmps(), voltage.toVolts(), voltage.toMarker())


class StatusPacket(object):
    def __init__(self, length, packetType, firmwareVersion, protocolVersion,
                 fineCurrent, coarseCurrent, voltage1, voltage2, outputVoltageSetting,
                 temperature, status, serialNumber, sampleRate, initialUsbVoltage,
                 initialAuxVoltage, hardwareRevision, eventCode, checkSum):
        self.length = length
        self.packetType = packetType
        self.firmwareVersion = firmwareVersion
        self.protocolVersion = protocolVersion
        self.fineCurrent = fineCurrent
        self.coarseCurrent = coarseCurrent
        self.voltage1 = voltage1
        self.voltage2 = voltage2
        self.outputVoltageSetting = outputVoltageSetting
        self.temperature = temperature
        self.status = status
        self.serialNumber = serialNumber
        self.sampleRate = sampleRate
        self.initialUsbVoltage = initialUsbVoltage
        self.initialAuxVoltage = initialAuxVoltage
        self.hardwareRevision = hardwareRevision
        self.eventCode = eventCode
        self.checkSum = checkSum


class Currents(object):
    def __init__(self, main, usb, aux):
        self.main = main
        self.usb = usb
        self.aux = aux


class BitReader(object):
    """ Interpreter for binary stream """

    def __init__(self, filename):
        try:
            self.filesize = os.stat(filename).st_size
            self.fin = (open(filename, "rb"))
        except Exception as e:
            sys.stderr.write(e)

    def close(self):
        """ Close binary stream """
        try:
            self.fin.close()
        except Exception as e:
            sys.stderr.write(e)

    def skipBytes(self, nbytes):
        """ Forward and ignore nbytes from current reading position
        """
        try:
            self.fin.seek(nbytes, os.SEEK_CUR)
        except Exception as e:
            sys.stderr.write(e)

    def readUInt8(self):
        """ Read and return unsigned 8-bit integer from input
        """
        try:
            # B = unsigned char
            val = struct.unpack('B', self.fin.read(1))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readInt8(self):
        """ Read and return signed 8-bit integer from input
        """
        try:
            # b = signed char
            val = struct.unpack('b', self.fin.read(1))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readFloat32(self):
        """ Read and return 32-bit floating-point number from input
        """
        try:
            # f = floating point (4 bytes)
            val = struct.unpack('f', self.fin.read(4))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readUInt16(self):
        """ Read and return unsigned 16-bit integer from input
        """
        try:
            # H = unsigned short (2 bytes)
            val = struct.unpack('H', self.fin.read(2))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readInt16(self):
        """ Read and return signed 16-bit integer from input
        """
        try:
            # h = short (2 bytes)
            val = struct.unpack('h', self.fin.read(2))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readUInt32(self):
        """ Read and return unsigned 32-bit integer from input
        """
        try:
            # I = unsigned int (4 bytes)
            val = struct.unpack('I', self.fin.read(4))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readInt32(self):
        """ Read and return signed 32-bit integer from input
        """
        try:
            # i = int (4 bytes)
            val = struct.unpack('i', self.fin.read(4))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readUInt64(self):
        """ Read and return unsigned 64-bit integer from input
        """
        try:
            # Q = unsigned long long (8 bytes)
            val = struct.unpack('Q', self.fin.read(8))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readInt64(self):
        """ Read and return signed 64-bit integer from input
        """
        try:
            # q = long long (8 bytes)
            val = struct.unpack('q', self.fin.read(8))[0]
        except Exception as e:
            sys.stderr.write(e)
        return val

    def readString(self, length):
        """ Read and return length-byte string from input
        """
        assert length >= 0

        try:
            s = self.fin.read(length)
            return s.strip()
        except Exception as e:
            sys.stderr.write(e)

        return None

    def isFinished(self):
        """ Return whether finished to read input
        """
        try:
            return self.fin.tell() == self.filesize
        except Exception as e:
            sys.stderr.write(e)
            self.fin.close()

        return True

    def readDateTime(self):
        """ Read and return date in epoch format
        """
        val = self.readUInt64()
        ticks = val & 0x3FFFFFFFFFFFFFFF
        s = ticks / 10.0 ** 7
        delta = datetime.timedelta(seconds=s)
        dt = datetime.datetime(1, 1, 1) + delta
        return dt


class Pt4FileReader(BitReader):
    """ Reader for Monsoon Power Monitor .pt4 file """

    def readHeader(self):
        """ Read and return header from Pt4 file
        """
        headerSize = self.readUInt32()
        identifier = self.readString(20)
        batterySize = self.readUInt32()
        captureDate = self.readDateTime()
        serial = self.readString(20)
        calibrationStatus = self.readUInt32()
        voutSetting = self.readUInt32()
        voutValue = self.readFloat32()
        hardwareRate = self.readUInt32()
        softwareRate = self.readFloat32()

        powerField = self.readUInt32()
        currentField = self.readUInt32()
        voltageField = self.readUInt32()

        captureSetting = self.readString(30)
        swVersion = self.readString(10)
        runMode = self.readUInt32()
        exitCode = self.readUInt32()
        totalCount = self.readInt64()

        statusOffset = self.readUInt16()
        statusSize = self.readUInt16()
        sampleOffset = self.readUInt16()
        sampleSize = self.readUInt16()

        initialMainVoltage = self.readUInt16()
        initialUsbVoltage = self.readUInt16()
        initialAuxVoltage = self.readUInt16()

        captureDataMask = self.readUInt16()

        sampleCount = self.readUInt64()
        missingCount = self.readUInt64()

        sumMainVoltage = self.readFloat32()
        sumMainCurrent = self.readFloat32()
        sumMainPower = self.readFloat32()

        sumUsbVoltage = self.readFloat32()
        sumUsbCurrent = self.readFloat32()
        sumUsbPower = self.readFloat32()

        sumAuxVoltage = self.readFloat32()
        sumAuxCurrent = self.readFloat32()
        sumAuxPower = self.readFloat32()

        # Padded to 272 bytes; status packet begins
        self.skipBytes(60)

        appInfo = ApplicationInfo(captureSetting, swVersion, runMode, exitCode)
        samples = Samples(captureDataMask, totalCount, statusOffset, statusSize,
                          sampleOffset, sampleSize)

        sumMain = SumValues(sumMainVoltage, sumMainCurrent, sumMainPower)
        sumUsb = SumValues(sumUsbVoltage, sumUsbCurrent, sumUsbPower)
        sumAux = SumValues(sumAuxVoltage, sumAuxCurrent, sumAuxPower)
        sumAll = Sum(sumMain, sumUsb, sumAux)

        return Header(headerSize, identifier, batterySize, captureDate, serial,
                      calibrationStatus, voutSetting, voutValue, hardwareRate,
                      softwareRate, powerField, currentField, voltageField, appInfo,
                      samples, sumAll)

    def readStatusPacket(self):
        """ Read and return status packet from input stream
        """
        length = self.readUInt8()
        packetType = self.readUInt8()
        firmwareVersion = self.readUInt8()
        protocolVersion = self.readUInt8()

        mainFineCurrent = self.readInt16()
        usbFineCurrent = self.readInt16()
        auxFineCurrent = self.readInt16()

        voltage1 = self.readUInt16()

        mainCoarseCurrent = self.readInt16()
        usbCoarseCurrent = self.readInt16()
        auxCoarseCurrent = self.readInt16()

        voltage2 = self.readUInt16()

        outputVoltageSetting = self.readUInt8()
        temperature = self.readUInt8()
        status = self.readUInt8()

        self.skipBytes(3)

        serialNumber = self.readUInt16()
        sampleRate = self.readUInt8()

        self.skipBytes(11)

        initialUsbVoltage = self.readUInt16()
        initialAuxVoltage = self.readUInt16()
        hardwareRevision = self.readUInt8()

        self.skipBytes(11)

        eventCode = self.readUInt8()

        self.skipBytes(2)

        checkSum = self.readUInt8()

        # padded to 1024, sample data begins
        self.skipBytes(692)

        fineCurrent = Currents(mainFineCurrent, usbFineCurrent, auxFineCurrent)
        coarseCurrent = Currents(mainCoarseCurrent, usbCoarseCurrent,
                                 auxCoarseCurrent)

        # Only suppor hardware revisions from 3+
        # Older revisions have different sample-data format
        if hardwareRevision < 3:
            sys.stderr.write("Old hardware revision ({0}) is not supported!".format(hardwareRevision))

        return StatusPacket(length, packetType, firmwareVersion,
                            protocolVersion, fineCurrent, coarseCurrent, voltage1,
                            voltage2, outputVoltageSetting, temperature, status,
                            serialNumber, sampleRate, initialUsbVoltage, initialAuxVoltage,
                            hardwareRevision, eventCode, checkSum)

    def readSample(self, header):
        """ Read and return sample data from input stream
        """
        mainCurrent = usbCurrent = auxCurrent = 0

        if (header.samples.captureDataMask & 0x1000 != 0):
            mainCurrent = self.readInt16()

        if (header.samples.captureDataMask & 0x2000 != 0):
            usbCurrent = self.readInt16()

        if (header.samples.captureDataMask & 0x4000 != 0):
            auxCurrent = self.readInt16()

        voltage = self.readUInt16()

        return RawSample(mainCurrent, usbCurrent, auxCurrent, voltage)

    @classmethod
    def readAsVector(cls, filename):
        reader = cls(filename)
        header = reader.readHeader()
        statusPacket = reader.readStatusPacket()
        seq = []
        count = 0

        while reader.isFinished() is False:
            rawSample = reader.readSample(header)
            sample = Sample.fromRaw(rawSample, statusPacket)
            # if count % 100 != 0:
            #     continue
            yield (header, statusPacket, count, sample)
            count += 1

        reader.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {0} <pt4-file>".format(sys.argv[0]))
        sys.exit(1)

    print("time,current,power,marker")
    for smpl in Pt4FileReader.readAsVector(sys.argv[1]):
        print('{:7.4f}'.format(smpl[2] * 0.0002) + ',' + str(smpl[3]))
