/*------------------------------------------------------------------------------
--         Copyright (c) 2020-2023, Inspur Inc. All rights reserved           --
--                                                                            --
-- This software is confidential and proprietary and may be used only as      --
--   expressly authorized by Inspur in a written licensing agreement.         --
--                                                                            --
-- Description:                                                               --
--   this is not a testcase, but a feature to download FW via PCIe to LMU     --
------------------------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#include "dma_utils.c"

#define I_AM_READY 0x11111111
#define DOWNLOADED 0x22222222
#define CHECK_PASS 0x33333333
#define CHECK_FAIL 0x44444444
#define HOST_RV_RES 0x55555555
#define FIRMWARE_HEADER_SIZE 64
#define FIRMWARE_HEADER_ADDR (0x40600000)
#define STATUS_ADDR (FIRMWARE_HEADER_ADDR + FIRMWARE_HEADER_SIZE)
#define STATUS_SIZE (sizeof(unsigned int))

/* #define SIM_ROMCODE */
#define DRIVER_TO_DEVICE_NAME_DEFAULT "/dev/xdma0_h2c_0"
#define DRIVER_FROM_DEVICE_NAME_DEFAULT "/dev/xdma0_c2h_0"
#ifdef SIM_ROMCODE
#define ROMCODE_FROM_HOST_NAME_DEFAULT "/dev/xdma0_h2c_1"
#define ROMCODE_TO_HOST_NAME_DEFAULT "/dev/xdma0_c2h_1"
#endif

typedef struct
{
    char *firmware_file;
} t_thrd_initial;

void delay_ms(unsigned int delay_in_ms)
{
    if (1)
    {
        if (delay_in_ms > 999)
            sleep(delay_in_ms / 1000);
    }
    else if (0)
    {
        usleep(1000 * delay_in_ms);
    }
    else if (0)
    {
        struct timeval delay_time;
        delay_time.tv_sec = delay_in_ms / 1000;
        delay_time.tv_usec = 1000 * (delay_in_ms % 1000);
        select(0, NULL, NULL, NULL, &delay_time);
    }
    else if (0)
    {
        struct timespec req;
        req.tv_sec = delay_in_ms / 1000;
        req.tv_nsec = 1000000 * (delay_in_ms % 1000);
        if (-1 == nanosleep(&req, NULL))
        {
            printf("nanosleep %d is not supported\n", delay_in_ms);
        }
    }
    else if (0)
    {
        unsigned int i_count = 0xfffff;
        unsigned int i = 0;
        unsigned int elapsed_ms = 0;
        struct timeval tv_start, tv_end;
        gettimeofday(&tv_start, NULL);
        do
        {
            while (i++ < i_count)
            {
                ;
            }

            gettimeofday(&tv_end, NULL);

            elapsed_ms = (tv_end.tv_sec - tv_start.tv_sec) * 1000 + (tv_end.tv_usec - tv_start.tv_usec) / 1000;
        } while (elapsed_ms < delay_in_ms);
    }

    printf("delay as %d ms elapsed\n", delay_in_ms);
    return;
}

static unsigned int driver_thread(void *p_thrd_init_data)
{
    ssize_t rc = 0;
    char *driver_to_device_name = DRIVER_TO_DEVICE_NAME_DEFAULT;
    char *driver_from_device_name = DRIVER_FROM_DEVICE_NAME_DEFAULT;
    int driver_to_device_id;
    int driver_from_device_id;
    unsigned int fw_addr = FIRMWARE_HEADER_ADDR;
    unsigned int pstatus = STATUS_ADDR;
    unsigned int status = 0;
    char *p_c_status = (char *)&status;
    FILE *fw;
    unsigned int length = 0;
    unsigned char *buf = NULL;
    int i = 0;

    unsigned int err_ret = 0;

    t_thrd_initial t_param = *((t_thrd_initial *)p_thrd_init_data);

    driver_to_device_id = open(driver_to_device_name, O_RDWR);
    if (driver_to_device_id < 0)
    {
        fprintf(stderr, "unable to open device %s, %d.\n",
                driver_to_device_name, driver_to_device_id);
        perror("open device");
        err_ret++;
        goto tc_err;
    }

    driver_from_device_id = open(driver_from_device_name, O_RDWR);
    if (driver_from_device_id < 0)
    {
        fprintf(stderr, "unable to open device %s, %d.\n",
                driver_from_device_name, driver_from_device_id);
        perror("open device");
        err_ret++;
        goto tc_err;
    }

    fw = fopen(t_param.firmware_file, "rb");
    fseek(fw, 0, SEEK_END);
    length = ftell(fw);
    fseek(fw, 0, SEEK_SET);
    if (length < (FIRMWARE_HEADER_SIZE + 1))
    {
        printf("\nERROR: fireware size should be 513 at least!!!\n");
        err_ret++;
        goto tc_err;
    }
    buf = malloc(length);
    if (buf == NULL)
    {
        printf("\nERROR: malloc failed\n");
        err_ret++;
        goto tc_err;
    }
    fread(buf, 1, length, fw);
    fclose(fw);

    //check ready
    printf("\nwaiting for \'I AM READY\' from boot rom\n");
    i = 0;
    while (status != I_AM_READY)
    {
        rc = read_to_buffer(driver_from_device_name, driver_from_device_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("I_AM_READY waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    printf("\n\'I AM READY\' comes, start downloading firmware header\n");

    //download firmware header
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, buf, FIRMWARE_HEADER_SIZE, fw_addr);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }
#if 0 //this will be confirmed by PCIe
    rc = read_to_buffer(driver_from_device_name, driver_from_device_id, buf_cmp, length, fw_addr);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }
    if (memcmp(buf, buf_cmp, length) != 0)
    {
        printf("PCIe DMA wirting data error, please check it!\n");
        err_ret++;
        goto tc_err;
    }
#endif

    status = DOWNLOADED;
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    printf("\ndownloading firmware header done, start checking\n");
    //check result
    while (status != CHECK_PASS && status != CHECK_FAIL)
    {
        rc = read_to_buffer(driver_from_device_name, driver_from_device_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
    }

    if (status == CHECK_FAIL)
    {
        printf("\nERROR: Firmware header decode failed\n");
        err_ret++;
        goto tc_err;
    }

    // ack result
    status = HOST_RV_RES;
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    printf("\nWaiting for firmware address!!!\n");
    i = 0;
    while (status == HOST_RV_RES)
    {
        rc = read_to_buffer(driver_from_device_name, driver_from_device_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("firmware address cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'Waiting for firmware address\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    rc = read_to_buffer(driver_from_device_name, driver_from_device_id, (char *)&fw_addr, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }
    printf("\nFirmware address is 0x%08x!!!\n", fw_addr);
    // ack result
    status = HOST_RV_RES;
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    printf("\nWaiting for firmware content ready signal!!!\n");
    i = 0;
    while (status != I_AM_READY)
    {
        rc = read_to_buffer(driver_from_device_name, driver_from_device_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("I_AM_READY waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    printf("\n\'I AM READY\' comes, start firmware content downloading\n");

    //download firmware
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, (buf + FIRMWARE_HEADER_SIZE), (length - FIRMWARE_HEADER_SIZE), fw_addr);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }
    status = DOWNLOADED;
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    printf("\ndownloading firmware content done, start checking\n");
    //check result
    while (status != CHECK_PASS && status != CHECK_FAIL)
    {
        rc = read_to_buffer(driver_from_device_name, driver_from_device_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
    }

    if (status == CHECK_FAIL)
    {
        printf("\nERROR: Firmware content decode failed\n");
        err_ret++;
        goto tc_err;
    }

    // ack result
    status = HOST_RV_RES;
    rc = write_from_buffer(driver_to_device_name, driver_to_device_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    printf("\nFirmware download and decode success\n");

tc_err:
    if (buf != NULL)
    {
        free(buf);
    }

    if (driver_from_device_id >= 0)
        close(driver_from_device_id);
    if (driver_to_device_id >= 0)
        close(driver_to_device_id);
    if (err_ret > 0)
        return 1;
    else
        return 0;
}

#ifdef SIM_ROMCODE
static unsigned int romcode_thread(void *p_thrd_init_data)
{
    ssize_t rc = 0;
    unsigned int err_ret = 0;
    char *romcode_from_host_name = ROMCODE_FROM_HOST_NAME_DEFAULT;
    char *romcode_to_host_name = ROMCODE_TO_HOST_NAME_DEFAULT;
    int romcode_from_host_id;
    int romcode_to_host_id;
    unsigned int header = FIRMWARE_HEADER_ADDR;
    unsigned int pstatus = STATUS_ADDR;
    unsigned int status = 0;
    char *p_c_status = (char *)&status;
    int i = 0;

    romcode_from_host_id = open(romcode_from_host_name, O_RDWR);
    if (romcode_from_host_id < 0)
    {
        fprintf(stderr, "unable to open device %s, %d.\n",
                romcode_from_host_name, romcode_from_host_id);
        perror("open device");
        err_ret++;
        goto tc_err;
    }

    romcode_to_host_id = open(romcode_to_host_name, O_RDWR);
    if (romcode_to_host_id < 0)
    {
        fprintf(stderr, "unable to open device %s, %d.\n",
                romcode_to_host_name, romcode_to_host_id);
        perror("open device");
        err_ret++;
        goto tc_err;
    }

    status = I_AM_READY;
    rc = write_from_buffer(romcode_from_host_name, romcode_from_host_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    i = 0;
    while (status != DOWNLOADED)
    {
        rc = read_to_buffer(romcode_to_host_name, romcode_to_host_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("DOWNLOADED waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    status = CHECK_PASS;
    rc = write_from_buffer(romcode_from_host_name, romcode_from_host_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    i = 0;
    while (status != HOST_RV_RES)
    {
        rc = read_to_buffer(romcode_to_host_name, romcode_to_host_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("HOST_RV_RES waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    status = FIRMWARE_HEADER_ADDR + 2 * FIRMWARE_HEADER_SIZE;
    rc = write_from_buffer(romcode_from_host_name, romcode_from_host_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    i = 0;
    while (status != HOST_RV_RES)
    {
        rc = read_to_buffer(romcode_to_host_name, romcode_to_host_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("HOST_RV_RES waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    status = I_AM_READY;
    rc = write_from_buffer(romcode_from_host_name, romcode_from_host_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    i = 0;
    while (status != DOWNLOADED)
    {
        rc = read_to_buffer(romcode_to_host_name, romcode_to_host_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("DOWNLOADED waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }

    status = CHECK_PASS;
    rc = write_from_buffer(romcode_from_host_name, romcode_from_host_id, p_c_status, STATUS_SIZE, pstatus);
    if (rc < 0)
    {
        err_ret++;
        goto tc_err;
    }

    i = 0;
    while (status != HOST_RV_RES)
    {
        rc = read_to_buffer(romcode_to_host_name, romcode_to_host_id, p_c_status, STATUS_SIZE, pstatus);
        if (rc < 0)
        {
            err_ret++;
            goto tc_err;
        }
        printf("HOST_RV_RES waiting cycle %d\n", i);
        delay_ms(1000);
        if (i >= 30)
        {
            printf("\nERROR: \'I AM READY\' doesn't come in 30 seconds\n");
            err_ret++;
            goto tc_err;
        }
        i++;
    }
    
tc_err:
    if (romcode_from_host_id >= 0)
        close(romcode_from_host_id);
    if (romcode_to_host_id >= 0)
        close(romcode_to_host_id);
    if (err_ret > 0)
        return 1;
    else
        return 0;
}
#endif

int main(int argc, char *argv[])
{
#ifdef SIM_ROMCODE
    pthread_t thrd_id[2];
    t_thrd_initial thrd_init_data[2];
    unsigned int thrd_exit_code[2];
#else
    pthread_t thrd_id[1];
    t_thrd_initial thrd_init_data[1];
    unsigned int thrd_exit_code[1];
#endif

    if (argc < 2)
    {
        printf("donot have firmware image, please checkout it!!!\n");
        return 1;
    }
    printf("firmware image name: %s\n", argv[1]);
    thrd_init_data[0].firmware_file = argv[1];
    pthread_create(&(thrd_id[0]), NULL, (void *)(driver_thread), (void *)(&thrd_init_data[0]));
#ifdef SIM_ROMCODE
    pthread_create(&(thrd_id[1]), NULL, (void *)(romcode_thread), (void *)(&thrd_init_data[1]));
#endif

    pthread_join(thrd_id[0], (void *)(&thrd_exit_code[0]));
#ifdef SIM_ROMCODE
    pthread_join(thrd_id[1], (void *)(&thrd_exit_code[1]));
#endif

    printf("exit code from driver_thread is %d\n", thrd_exit_code[0]);
#ifdef SIM_ROMCODE
    printf("exit code from romcode_thread is %d\n", thrd_exit_code[1]);
#endif

    if (thrd_exit_code[0]
#ifdef SIM_ROMCODE
        || thrd_exit_code[1])
#else
    )
#endif
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
