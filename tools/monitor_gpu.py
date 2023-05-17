import GPUtil

def GetGpuInfo():
    gpulist = []
    # GPUtil.showUtilization()

    # 获取多个GPU的信息，存在列表里
    Gpus = GPUtil.getGPUs()
    for gpu in Gpus:
        # print('gpu.id:', gpu.id)
        # print('GPU总量：', gpu.memoryTotal)
        # print('GPU使用量：', gpu.memoryUsed)
        # print('gpu使用占比:', gpu.memoryUtil * 100)
        # 按GPU逐个添加信息

        gpu_memoryTotal = round((gpu.memoryTotal) / 1024)
        gpu.memoryUsed = round((gpu.memoryUsed) / 1024, 2)
        gpu_memoryUtil = round((gpu.memoryUtil) * 100, 2)
        gpulist.append([gpu.id, gpu_memoryTotal, gpu.memoryUsed, gpu_memoryUtil])  # GPU序号，GPU总量，GPU使用量，gpu使用占比
    print("GPU信息(G)：GPU序号，GPU总量，GPU使用量，gpu使用占比")
    return gpulist