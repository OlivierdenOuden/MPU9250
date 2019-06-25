#***************************************************************#
# 							                                   	#
#	SPI library for MPU9250	                  			     	#
# 					                                 			#
#	Olivier den Ouden & Corne Oudshoorn	                  		#
#	Royal Netherlands Meteorological Institute            		#
#	RDSA & RDWD				                               		#
#	Feb. 2019		                               				#
#				                                   				#
#***************************************************************#

#Modules
import spidev
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
import argparse
from argparse import RawTextHelpFormatter
import obspy
from obspy import Stream,Trace

# Parser arguments
parser = argparse.ArgumentParser(prog='MPU9250 sensor Read-out',
    description=('Read-out of a 9-axis mpu sensor\n'
    ), formatter_class=RawTextHelpFormatter
)

parser.add_argument(
    '-t', action='store', default=256, type=float,
    help='Time of recording, [sec], always power of 2!.\n', metavar='-time')

parser.add_argument(
    '-fs', action='store', default=1, type=float,
    help='Sample rate, [Hz].\n', metavar='-SamplFreq')

parser.add_argument(
    '-miniSeed', action='store', default=64, type=float,
    help='Saving the data after x [sec], always power of 2!.\n', metavar='-miniSeedFreq')

args = parser.parse_args()

# Time knowledge
Fs = args.fs
record_t = args.t
n_samples = record_t*Fs

delta_time = args.miniSeed
delta_sampl= delta_time*Fs

# Save data
Time_array = np.linspace(0,record_t,n_samples)
x = np.zeros((delta_sampl,))
y = np.zeros((delta_sampl,))
z = np.zeros((delta_sampl,))
# Header Channel type
stats_x = {'network': 'MBA_PI', 'station': '08_MPU',
         'channel': 'AC_x', 'npts': delta_sampl, 'sampling_rate': Fs,
         'mseed': {'dataquality': 'D'}}

stats_y = {'network': 'MBA_PI', 'station': '08_MPU',
         'channel': 'AC_y', 'npts': delta_sampl, 'sampling_rate': Fs,
         'mseed': {'dataquality': 'D'}}

stats_z = {'network': 'MBA_PI', 'station': '08_MPU',
         'channel': 'AC_z', 'npts': delta_sampl, 'sampling_rate': Fs,
         'mseed': {'dataquality': 'D'}}

''' MPU-9250 Register Addresses '''
SELF_TEST_X_GYRO_REG  = 0x00
SELF_TEST_Y_GYRO_REG  = 0x01
SELF_TEST_Z_GYRO_REG  = 0x02

SELF_TEST_X_ACCEL_REG = 0x0D
SELF_TEST_Y_ACCEL_REG = 0x0E
SELF_TEST_Z_ACCEL_REG = 0x0F

XG_OFFSET_H_REG       = 0x13
XG_OFFSET_L_REG       = 0x14
YG_OFFSET_H_REG       = 0x15
YG_OFFSET_L_REG       = 0x16
ZG_OFFSET_H_REG       = 0x17
ZG_OFFSET_L_REG       = 0x18
SMPLRT_DIV_REG        = 0x19
CONFIG_REG            = 0x1A
GYRO_CONFIG_REG       = 0x1B
ACCEL_CONFIG_REG      = 0x1C
ACCEL_CONFIG_2_REG    = 0x1D
LP_ACCEL_ODR_REG      = 0x1E
WOM_THR_REG           = 0x1F

FIFO_EN_REG           = 0x23

INT_PIN_CFG_REG       = 0x37
INT_ENABLE_REG        = 0x38

INT_STATUS_REG        = 0x3A
ACCEL_XOUT_H_REG      = 0x3B
ACCEL_XOUT_L_REG      = 0x3C
ACCEL_YOUT_H_REG      = 0x3D
ACCEL_YOUT_L_REG      = 0x3E
ACCEL_ZOUT_H_REG      = 0x3F
ACCEL_ZOUT_L_REG      = 0x40
TEMP_OUT_H_REG        = 0x41
TEMP_OUT_L_REG        = 0x42
GYRO_XOUT_H_REG       = 0x43
GYRO_XOUT_L_REG       = 0x44
GYRO_YOUT_H_REG       = 0x45
GYRO_YOUT_L_REG       = 0x46
GYRO_ZOUT_H_REG       = 0x47
GYRO_ZOUT_L_REG       = 0x48

SIGNAL_PATH_RESET_REG = 0x68
MOT_DETECT_CTRL_REG   = 0x69
USTER_CTRL_REG        = 0x6A
PWR_MGMT_1_REG        = 0x6B
PWR_MGMT_2_REG        = 0x6C

FIFO_COUNT_H_REG      = 0x72
FIFO_COUNT_L_REG      = 0x73
FIFO_R_W_REG          = 0x74
WHO_AM_I_REG          = 0x75

XA_OFFSET_H_REG       = 0x77
XA_OFFSET_L_REG       = 0x78

YA_OFFSET_H_REG       = 0x7A
YA_OFFSET_L_REG       = 0x7B

ZA_OFFSET_H_REG       = 0x7D
ZA_OFFSET_L_REG       = 0x7E

''' READ / WRITE '''
READ  = 0x80
WRITE = 0x00

''' SPI Bus Values '''
SPI_BUS   = 0
SPI_CS    = 0
SPI_SPEED = int(1e6)
SPI_MODE  = 0

class MPU9250(object):
    def __init__(self, bus=SPI_BUS, cs=SPI_CS, speed=SPI_SPEED, mode=SPI_MODE):
        self.bus = spidev.SpiDev()
        self.bus.open(bus, cs)
        self.bus.max_speed_hz = speed
        self.bus.mode = mode

    def __del__(self):
        self.bus.close()

    def config_accel(self, f_s, accel_config_2_val=0b00001010):
        valid_fs = [2, 4, 8, 16]

        if f_s not in valid_fs:
            raise ValueError("Value for  fs not allowed. Should be 2, 4, 8, or 16 g.")

        self.fs_accel = 16

        accel_config_val = valid_fs.index(f_s) << 3
        self.write(ACCEL_CONFIG_REG, accel_config_val)
        # default value: BW=99Hz, rate=1kHz
        self.write(ACCEL_CONFIG_2_REG, accel_config_2_val)

    def config_gyro(self,fs, config_reg_value=0b00000010):
        valid_fs = [250, 500, 1000, 2000]

        if fs not in valid_fs:
            raise ValueError("Value for fs not allowed. Should be 250, 500, 1000, or 1000 dps.")

        self.fs_gyro = fs

        # default BW=92Hz, FS=1kHz
        self.write(CONFIG_REG, config_reg_value)
        gyro_config_val = valid_fs.index(fs) << 3
        self.write(GYRO_CONFIG_REG, gyro_config_val)

    def config_magn(self):
        # to be written
        # peripheral i2c device has to be configured
        # because magnetometer is separate i2c device
        pass

    def write(self, register, value):
        mosi = [WRITE | register, value]
        self.bus.xfer2(mosi)

    def read(self, register, n_regs=1):
        mosi = [READ | register] # opcode
        # add "zero" bytes for additonal clock cycles
        # to read <n_regs> registers
        mosi.extend([0x00] * n_regs)
        miso = self.bus.xfer2(mosi)
        data = miso[1::]

        return data

    def uint_to_int(self, data):
        if len(data) != 2:
            raise ValueError("This function can only convert a two byte integer.")

        value = data[0] << 8 | data[1]
        # make negative if MSB equals 1
        if value & (1 << (16 - 1)):
            value -= 1 << 16

        return value

    def read_xyz_accel(self):
        data = self.read(ACCEL_XOUT_H_REG, 6)
        x_value = self.fs_accel / (1 << (16 - 1)) * self.uint_to_int(data[0:2])
        y_value = self.fs_accel / (1 << (16 - 1)) * self.uint_to_int(data[2:4])
        z_value = self.fs_accel / (1 << (16 - 1)) * self.uint_to_int(data[4:6])
	
        return {'x': x_value, 'y': y_value, 'z': z_value}

    def read_xyz_gyro(self):
        data = self.read(GYRO_XOUT_H_REG, 6)
        x_value = self.fs_gyro / (1 << (16 - 1)) * self.uint_to_int(data[0:2])
        y_value = self.fs_gyro / (1 << (16 - 1)) * self.uint_to_int(data[2:4])
        z_value = self.fs_gyro / (1 << (16 - 1)) * self.uint_to_int(data[4:6])

        return {'x': x_value, 'y': y_value, 'z': z_value}

    def read_xyz_magn(self):
        pass

    def read_temp(self):
        data = self.read(TEMP_OUT_H_REG, 2)
        temp_value = (data[0] << 8 | data[1]) / 333.87 + 21 

        return temp_value

if __name__ == "__main__":
	print('')
	print('MPU9250 sensor Read-out')
	print('')
	print('Olivier den Ouden & Corne Oudshoorn')
	print('Royal Netherlands Meteorological Institute, KNMI')
	print('Feb. 2019')
	print('')

	i = 0
	j = 0
	while i < n_samples:
		if j == delta_sampl:
			stats['starttime'] = st
			sX = Stream([Trace(data=x[:,], header=stats_x)])
			sY = Stream([Trace(data=y[:,], header=stats_y)])
			sZ = Stream([Trace(data=z[:,], header=stats_z)])			
			sX.write("MPU9250_Accl_X_"+str(st_str)+".mseed", format='MSEED', reclen=512)
			sY.write("MPU9250_Accl_Y_"+str(st_str)+".mseed", format='MSEED', reclen=512)
			sZ.write("MPU9250_Accl_Z_"+str(st_str)+".mseed", format='MSEED', reclen=512)
			sPres.write("LPS33HW_Pres_"+str(st_str)+".mseed", format='MSEED', reclen=512)
			shutil.move("/home/pi/PIM_scripts/SPI/mpu9250_spi_lib/MPU9250_Accl_X_"+str(st_str)+".mseed","/home/pi/PIM_data/mseed/MPU9250_Accl_X__"+str(st_str)+".mseed")
			shutil.move("/home/pi/PIM_scripts/SPI/mpu9250_spi_lib/MPU9250_Accl_Y_"+str(st_str)+".mseed","/home/pi/PIM_data/mseed/MPU9250_Accl_Y__"+str(st_str)+".mseed")			
			shutil.move("/home/pi/PIM_scripts/SPI/mpu9250_spi_lib/MPU9250_Accl_Z_"+str(st_str)+".mseed","/home/pi/PIM_data/mseed/MPU9250_Accl_Z__"+str(st_str)+".mseed")

			x = np.zeros((delta_sampl,))
			y = np.zeros((delta_sampl,))
			z = np.zeros((delta_sampl,))
			j = 0

		if j==0:
			st = datetime.now()
			st_str = st.strftime('%Y_%m_%d_T%H_%M_%S.%f')


		mpu = MPU9250()
		mpu.config_accel(16)
		data = mpu.read_xyz_accel()
		print(data)		
		x[j,] = data['x']
		y[j,] = data['y']
		z[j,] = data['z']
		
		j = j+1
		i = i+1

        # Print converted data
		print(data['x'])

		# Sampling rate
		time.sleep(1/Fs)

