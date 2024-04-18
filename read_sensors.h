#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "BusCommunication.h"

namespace rs{
	//传入地址（imu存的），对内存位置修改
	//调用的时候选从那个字节开始

	float a, b, c, d, e, f;
	float* get_F_x = &(a);
	float* get_F_y = &(b);
	float* get_F_z = &(c);

	float* get_M_x = &(d);
	float* get_M_y = &(e);
	float* get_M_z = &(f);

	int read_sensors(float * get_sensor, int start_byte)
	{
		InitMaster(26, 6, 0); //processimage里面的input和output分别是多少

		GetMasterPdo(0, start_byte, 32, get_F_x);
		GetMasterPdo(0, start_byte + 32, 32, get_F_y);
		GetMasterPdo(0, start_byte + 64, 32, get_F_z);
		GetMasterPdo(0, start_byte + 96, 32, get_M_x);
		GetMasterPdo(0, start_byte + 128, 32, get_M_y);
		GetMasterPdo(0, start_byte + 160, 32, get_M_z);

		get_sensor[0] = *get_F_x;
		get_sensor[1] = *get_F_y;
		get_sensor[2] = *get_F_z;
		get_sensor[3] = *get_M_x;
		get_sensor[4] = *get_M_y;
		get_sensor[5] = *get_M_z;
		
		return 0;
	}

}