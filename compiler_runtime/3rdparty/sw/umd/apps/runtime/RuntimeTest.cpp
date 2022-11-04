/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Inspur.
 * This is a new or modified file.
 */

#include "DlaImageUtils.h"
#include "ErrorMacros.h"
#include "RuntimeTest.h"

#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#include "half.h"
#include "main.h"
#include "nvdla_os_inf.h"

#include "dlaerror.h"
#include "dlatypes.h"

#include <cstdio> // snprintf, fopen
#include <string>

#define OUTPUT_DIMG "output.dimg"

using namespace half_float;

static TestImageTypes getImageType(std::string imageFileName)
{
    TestImageTypes it = IMAGE_TYPE_UNKNOWN;
    std::string ext = imageFileName.substr(imageFileName.find_last_of(".") + 1);
    if (ext == "pgm")
    {
        it = IMAGE_TYPE_PGM;
    }
    else if (ext == "jpg")
    {
        it = IMAGE_TYPE_JPG;
    }

    return it;
}

static NvDlaError copyImageToInputTensor
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pImgBuffer,
    nvdla::IRuntime::NvDlaTensor *tensorDesc
)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IRuntime::AipuConfig *aipu_config = i->aipu_config;
    NvDlaImage* R8Image = new NvDlaImage();
    NvDlaImage* tensorImage = NULL;
    if (!R8Image)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    if (aipu_config->mac_is_first_task) {
        R8Image->m_meta.channel = tensorDesc->dims.c;
        R8Image->m_meta.height = tensorDesc->dims.h;
        R8Image->m_meta.width = tensorDesc->dims.w;
        R8Image->m_meta.lineStride = tensorDesc->stride[1];
        R8Image->m_meta.surfaceStride = tensorDesc->stride[2];
        R8Image->m_meta.size = tensorDesc->bufferSize;
        R8Image->m_meta.surfaceFormat = NvDlaImage::TVMArray;

        R8Image->m_pData = aipu_config->image_addr;
        
    } else {
        std::string imgPath = /*i->inputImagesPath + */appArgs->inputName;
        TestImageTypes imageType = getImageType(imgPath);
        NvDlaDebugPrintf("In %s : %d, Image type: %d", __func__, __LINE__, IMAGE_TYPE_PGM);
        switch(imageType) {
            case IMAGE_TYPE_PGM:
                PROPAGATE_ERROR(PGM2DIMG(imgPath, R8Image, tensorDesc));
                break;
            case IMAGE_TYPE_JPG:
                PROPAGATE_ERROR(JPEG2DIMG(imgPath, R8Image, tensorDesc));
                break;
            default:
                //TODO Fix this error condition
    //          ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown image type: %s", imgPath.c_str());
                NvDlaDebugPrintf("Unknown image type: %s", imgPath.c_str());
                goto fail;
        }
    }

    tensorImage = i->inputImage;
    if (tensorImage == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL input Image");

    PROPAGATE_ERROR(createImageCopy(appArgs, R8Image, tensorDesc, tensorImage));
    NvDlaDebugPrintf("\nCreate image copy done !\n");
    //tensorImage->printBuffer(true);  /* Print the input Buffer */ 
    if (!aipu_config->mac_is_first_task)
        PROPAGATE_ERROR(DIMG2DlaBuffer(tensorImage, pImgBuffer));
    else {
        ssize_t rc = NvDlaWrite(*pImgBuffer, tensorImage->m_pData, tensorImage->m_meta.size);
        if (rc != tensorImage->m_meta.size) 
            NvDlaDebugPrintf("Write Image Error, size should be %d, but write %d.\n", tensorImage->m_meta.size, rc);
    }

    NvDlaDebugPrintf("\nimage write done !\n");
fail:
    if (R8Image != NULL && R8Image->m_pData != NULL && aipu_config == NULL)
        NvDlaFree(R8Image->m_pData);
    NvDlaDebugPrintf("\nimage delet !\n");
    delete R8Image;

    return e;
}

static NvDlaError prepareOutputTensor
(
    nvdla::IRuntime::NvDlaTensor* pTDesc,
    NvDlaImage* pOutImage,
    void** pOutBuffer,
    const TestAppArgs* appArgs,
    const TestInfo* appInfo
)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IRuntime::AipuConfig *aipu_config = appInfo->aipu_config;
    ssize_t rc;
    PROPAGATE_ERROR_FAIL(Tensor2DIMG(appArgs, pTDesc, pOutImage));
    if (aipu_config == NULL)
        PROPAGATE_ERROR_FAIL(DIMG2DlaBuffer(pOutImage, pOutBuffer));
    else {
        rc = NvDlaWrite(*pOutBuffer, pOutImage->m_pData, pOutImage->m_meta.size);
        if (rc != pOutImage->m_meta.size) 
            NvDlaDebugPrintf("Write Image Error, size should be %d, but write %d.\n", pOutImage->m_meta.size, rc);
    }
fail:
    return e;
}


NvDlaError setupInputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pInputBuffer
)
{
    NvDlaError e = NvDlaSuccess;
    nvdla::IRuntime::AipuConfig *aipu_config = i->aipu_config;
    void *hMem = NULL;
    nvdla::IRuntime::NvDlaTensor tDesc;
    nvdla::IRuntime* runtime = i->runtime;
    PROPAGATE_ERROR_FAIL(runtime->getInputTensorDesc(0, &tDesc));
    /*
    NvDlaDebugPrintf("In %s : %d, Get MAC task InputTensorDesc:\n"
            "warning: IF INPUT IS IMAGE, SHAPE SHOULD BE SAME WITH TVMArray.\n"
		    "\tinput shape is %d x %d x %d x %d,\n"
		    "\tinput stride is %d x %d,\n"
		    "\tbuffer size is %d,\n",
		    __func__, __LINE__,
		    tDesc.dims.n, tDesc.dims.h, tDesc.dims.w, tDesc.dims.c,
		    tDesc.stride[0], tDesc.stride[1],
		    tDesc.bufferSize); 
    */
    if ((aipu_config == NULL) || (aipu_config->mac_is_first_task)) {
        NvS32 numInputTensors = 0;
        if (!runtime)
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

        PROPAGATE_ERROR_FAIL(runtime->getNumInputTensors(&numInputTensors));

        i->numInputs = numInputTensors;

        if (numInputTensors < 1)
            goto fail;

        if (aipu_config == NULL) {
            PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pInputBuffer));
        } else {//aipu_mode && mac is first task
            nvdla::IRuntime::MemoryHandle *hMem_tmp = (nvdla::IRuntime::MemoryHandle *)malloc(sizeof(nvdla::IRuntime::MemoryHandle));
            hMem_tmp->input_addr = aipu_config->input_addr[0];
            hMem_tmp->base_addr = aipu_config->base_addr;
            pInputBuffer = (void **)(&(hMem_tmp->input_addr));//Do not use base_addr and addr_offset
            hMem = (void *)hMem_tmp;
            //NvDlaDebugPrintf("In %s : %d, set input buffer address is 0x%08x.\n", __func__, __LINE__, aipu_config->input_addr[0]);
        }
        i->inputHandle = (NvU8 *)hMem;
        PROPAGATE_ERROR_FAIL(copyImageToInputTensor(appArgs, i, pInputBuffer, &tDesc));
        if (!runtime->bindInputTensor(0,aipu_config == NULL ? false : true, hMem))
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindInputTensor() failed");
    } else {// aipu_mode && mac is not first task
            nvdla::IRuntime::MemoryHandle *hMem_tmp = (nvdla::IRuntime::MemoryHandle *)malloc(sizeof(nvdla::IRuntime::MemoryHandle));
            hMem_tmp->input_addr = aipu_config->input_addr[0];
            hMem_tmp->base_addr = aipu_config->base_addr;
            pInputBuffer = (void **)(&(hMem_tmp->input_addr));//Do not use base_addr and addr_offset
            hMem = (void *)hMem_tmp;
            i->inputHandle = (NvU8 *)hMem;
            //NvDlaDebugPrintf("In %s : %d, set input buffer address is 0x%08x.\n", __func__, __LINE__, aipu_config->input_addr[0]);
            if (!runtime->bindInputTensor(0,aipu_config == NULL ? false : true, hMem))
                ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindInputTensor() failed");

            for (NvU8 i = 0; i < (aipu_config->input_num - 1); i++) {
                hMem_tmp = (nvdla::IRuntime::MemoryHandle *)malloc(sizeof(nvdla::IRuntime::MemoryHandle));
                hMem_tmp->input_addr = aipu_config->input_addr[i + 1];
                hMem_tmp->base_addr = aipu_config->base_addr;
                hMem = (void *)hMem_tmp;
                //NvDlaDebugPrintf("In %s : %d, set other input buffer address is 0x%08x.\n", __func__, __LINE__, aipu_config->input_addr[i + 1]);
                if (!runtime->bindInputTensor(i + 1,aipu_config == NULL ? false : true, hMem))
                    ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindInputTensor() failed");
                
            }
    }

fail:
    return e;
}

static void cleanupInputBuffer(const TestAppArgs *appArgs,
                                TestInfo *i)
{
    nvdla::IRuntime *runtime = NULL;
    NvS32 numInputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaError e = NvDlaSuccess;
    if (i->inputImage != NULL && i->inputImage->m_pData != NULL) {
        NvDlaFree(i->inputImage->m_pData);
        i->inputImage->m_pData = NULL;
    }

    runtime = i->runtime;
    if (runtime == NULL)
        return;
    e = runtime->getNumInputTensors(&numInputTensors);
    if (e != NvDlaSuccess)
        return;

    if (numInputTensors < 1)
        return;

    e = runtime->getInputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess)
        return;

    if (i->inputHandle == NULL)
        return;

    /* Free the buffer allocated */
    if (i->aipu_config == NULL) {
        runtime->freeSystemMemory(i->inputHandle, tDesc.bufferSize);
    } else {
        free(i->inputHandle);
    }
    i->inputHandle = NULL;
    return;
}

NvDlaError setupOutputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pOutputBuffer
)
{
    NVDLA_UNUSED(appArgs);

    NvDlaError e = NvDlaSuccess;
    void *hMem = NULL;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaImage *pOutputImage = NULL;
    
    nvdla::IRuntime* runtime = i->runtime;
    nvdla::IRuntime::AipuConfig *aipu_config = i->aipu_config;

    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    PROPAGATE_ERROR_FAIL(runtime->getNumOutputTensors(&numOutputTensors));

    i->numOutputs = numOutputTensors;

    if (numOutputTensors < 1)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Expected number of output tensors of %u, found %u", 1, numOutputTensors);

    PROPAGATE_ERROR_FAIL(runtime->getOutputTensorDesc(0, &tDesc));
    
    if (aipu_config == NULL) {
        PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pOutputBuffer));
        i->outputHandle = (NvU8 *)hMem;
        
        pOutputImage = i->outputImage;
        if (i->outputImage == NULL)
            ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL Output image");
    
        PROPAGATE_ERROR_FAIL(prepareOutputTensor(&tDesc, pOutputImage, pOutputBuffer, appArgs, i));
    } else {//aipu_mode && mac is first task
        nvdla::IRuntime::MemoryHandle *hMem_tmp = (nvdla::IRuntime::MemoryHandle *)malloc(sizeof(nvdla::IRuntime::MemoryHandle));
        hMem_tmp->output_addr = aipu_config->output_addr;
        hMem_tmp->base_addr = aipu_config->base_addr;
        pOutputBuffer = (void **)(&(hMem_tmp->output_addr));
        hMem = (void *)hMem_tmp;
        i->outputHandle = (NvU8 *)hMem;
        //NvDlaDebugPrintf("In %s, set output buffer address is 0x%08x.\n", __func__, aipu_config->output_addr);
        /* PROPAGATE_ERROR_FAIL(prepareOutputTensor(&tDesc, pOutputImage, pOutputBuffer, appArgs, i)); */
    }

    if (!runtime->bindOutputTensor(0, aipu_config == NULL ? false : true, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindOutputTensor() failed");

fail:
    return e;
}

static void cleanupOutputBuffer(const TestAppArgs *appArgs,
                                TestInfo *i)
{
    nvdla::IRuntime *runtime = NULL;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaError e = NvDlaSuccess;
    /* Do not clear outputImage if in server mode */
    if (!i->dlaServerRunning &&
            i->outputImage != NULL &&
            i->outputImage->m_pData != NULL) {
        NvDlaFree(i->outputImage->m_pData);
        i->outputImage->m_pData = NULL;
    }

    runtime = i->runtime;
    if (runtime == NULL)
        return;
    e = runtime->getNumOutputTensors(&numOutputTensors);
    if (e != NvDlaSuccess)
        return;
    e = runtime->getOutputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess)
        return;

    if (i->outputHandle == NULL)
        return;

    /* Free the buffer allocated */
    if (i->aipu_config == NULL) {
        runtime->freeSystemMemory(i->outputHandle, tDesc.bufferSize);
    }
    i->outputHandle = NULL;
    return;
}

static NvDlaError readLoadable(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(appArgs);
    std::string loadableName;
    NvDlaFileHandle file;
    NvDlaStatType finfo;
    size_t file_size;
    NvU8 *buf = 0;
    size_t actually_read = 0;
    NvDlaError rc;

    // Determine loadable path
    if (appArgs->loadableName == "")
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No loadable found to load");
    }

    loadableName = appArgs->loadableName;

    rc = NvDlaFopen(loadableName.c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "couldn't open %s\n", loadableName.c_str());
    }

    rc = NvDlaFstat(file, &finfo);
    if ( rc != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "couldn't get file stats for %s\n", loadableName.c_str());
    }

    file_size = NvDlaStatGetSize(&finfo);
    if ( !file_size ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "zero-length for %s\n", loadableName.c_str());
    }

    buf = new NvU8[file_size];

    NvDlaFseek(file, 0, NvDlaSeek_Set);

    rc = NvDlaFread(file, buf, file_size, &actually_read);
    if ( rc != NvDlaSuccess )
    {
        NvDlaFree(buf);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read error for %s\n", loadableName.c_str());
    }
    NvDlaFclose(file);
    if ( actually_read != file_size ) {
        NvDlaFree(buf);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read wrong size for buffer? %d\n", actually_read);
    }

    i->pData = buf;

fail:
    return e;
}

NvDlaError loadLoadable(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    if (!runtime->load(i->pData, 0, i->aipu_config))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->load failed");

fail:
    return e;
}

void unloadLoadable(const TestAppArgs* appArgs, TestInfo *i)
{
    NVDLA_UNUSED(appArgs);
    nvdla::IRuntime *runtime = NULL;

    runtime = i->runtime;
    if (runtime != NULL) {
        runtime->unload(i->aipu_config);
    }
}

double get_elapsed_time(struct timespec *before, struct timespec *after)
{
  double deltat_s  = (after->tv_sec - before->tv_sec) * 1000000;
  double deltat_ns = (after->tv_nsec - before->tv_nsec) / 1000;
  return deltat_s + deltat_ns;
}

NvDlaError runTest(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    void* pInputBuffer = NULL;
    void* pOutputBuffer = NULL;
    /* struct timespec before, after; */
    nvdla::IRuntime::AipuConfig *aipu_config = i->aipu_config;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");
    if ((aipu_config == NULL) || (aipu_config->mac_is_first_task)) {
        i->inputImage = new NvDlaImage();
        i->outputImage = new NvDlaImage();
    }

    PROPAGATE_ERROR_FAIL(setupInputBuffer(appArgs, i, &pInputBuffer));
    if (aipu_config->net_parse_done) {
        goto parse_done;
    }
    PROPAGATE_ERROR_FAIL(setupOutputBuffer(appArgs, i, &pOutputBuffer));
    //NvDlaDebugPrintf("submitting tasks...\n");
    /* clock_gettime(CLOCK_MONOTONIC, &before); */
    if (!runtime->submit(aipu_config))
        ORIGINATE_ERROR(NvDlaError_BadParameter, "runtime->submit() failed");

    /* clock_gettime(CLOCK_MONOTONIC, &after); */
    /* NvDlaDebugPrintf("execution time = %f s\n", get_elapsed_time(&before,&after)); */
    if (aipu_config == NULL) {
        PROPAGATE_ERROR_FAIL(DlaBuffer2DIMG(&pOutputBuffer, i->outputImage));
        //i->outputImage->printBuffer(true);   /* Print the output buffer */
        /* Dump output dimg to a file */
        PROPAGATE_ERROR_FAIL(DIMG2DIMGFile(i->outputImage, OUTPUT_DIMG, true, appArgs->rawOutputDump));
    /*
    } else {
        ssize_t rc = NvDlaWrite((void *)i->outputImage->m_pData, (void *)(pOutputBuffer), 
                                i->outputImage->m_meta.size);
        if (rc != i->outputImage->m_meta.size)
            NvDlaDebugPrintf("Error: DlaBuffer2DIMG fail.\n");
        PROPAGATE_ERROR_FAIL(DIMG2DIMGFile(i->outputImage, OUTPUT_DIMG, true, appArgs->rawOutputDump));
    */
    }


fail:
    cleanupOutputBuffer(appArgs, i);
    /* Do not clear outputImage if in server mode */
    if (!i->dlaServerRunning && i->outputImage != NULL) {
        delete i->outputImage;
        i->outputImage = NULL;
    }
parse_done:
    cleanupInputBuffer(appArgs, i);
    if (i->inputImage != NULL) {
        delete i->inputImage;
        i->inputImage = NULL;
    }

    return e;
}

NvDlaError run(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    //AipuConfig aipu_config = AipuConfig();
    /* Create runtime instance */
    NvDlaDebugPrintf("creating new runtime context...\n");
    i->runtime = nvdla::createRuntime();
    if (i->runtime == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createRuntime() failed");

    if (!i->dlaServerRunning)
        PROPAGATE_ERROR_FAIL(readLoadable(appArgs, i));

    /* Load loadable */
    PROPAGATE_ERROR_FAIL(loadLoadable(appArgs, i));

    /* Start emulator */
    if (!i->runtime->initEMU())
        ORIGINATE_ERROR(NvDlaError_DeviceNotFound, "runtime->initEMU() failed");

    /* Run test */
    PROPAGATE_ERROR_FAIL(runTest(appArgs, i));

fail:
    /* Stop emulator */
    if (i->runtime != NULL)
        i->runtime->stopEMU();

    /* Unload loadables */
    unloadLoadable(appArgs, i);

    /* Free if allocated in read Loadable */
    if (!i->dlaServerRunning && i->pData != NULL) {
        delete[] i->pData;
        i->pData = NULL;
    }

    /* Destroy runtime */
    nvdla::destroyRuntime(i->runtime);

    return e;
}

NvDlaError mac_task_parse(nvdla::IRuntime::AipuConfig *aipu_config)
{
    //printf("In %s : %d, Start parse mac task.\n", __func__, __LINE__);
    NvDlaError e = NvDlaSuccess;
    TestInfo mac_task_info = TestInfo();
    /* readLoadable */
    void *loadable_tmp = (void *)malloc(aipu_config->loadable.second);
    memcpy(loadable_tmp, (void *)aipu_config->loadable.first, (size_t)aipu_config->loadable.second);
    /* mac_task_info.pData = aipu_config->loadable.first; */
    mac_task_info.pData = (NvU8 *)loadable_tmp;
    mac_task_info.aipu_config = aipu_config;
    mac_task_info.runtime = nvdla::createRuntime();

    TestAppArgs mac_task_args = TestAppArgs();
    mac_task_args.inputName = "TVMArray";
    /* Load loadable */
    PROPAGATE_ERROR_FAIL(loadLoadable(NULL, &mac_task_info));
    PROPAGATE_ERROR_FAIL(runTest(&mac_task_args, &mac_task_info));
fail:
    /* Unload loadables */
    unloadLoadable(&mac_task_args, &mac_task_info);

    /* Free if allocated in read Loadable */
    if (mac_task_info.pData != NULL) {
        delete[] mac_task_info.pData;
        mac_task_info.pData = NULL;
    }

    /* Destroy runtime */
    nvdla::destroyRuntime(mac_task_info.runtime);

    return e;
}
