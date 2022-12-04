import argparse
import os
import os.path as osp
import re
from glob import glob

import pandas as pd


def to_milli_second(time: str):
    t = 0.0
    if re.search('^.*\ds$', time) is not None:
        t = float(time[:-1]) * 1000
    elif re.search('^.*\dms$', time) is not None:
        t = float(time[:-2])
    elif re.search('^.*\dus$', time) is not None:
        t = float(time[:-2]) / 1000
    return t


def get_mem_copy(line_cuda_mem, cp_type):
    line_cuda_mem = line_cuda_mem.replace('CUDA memcpy {}'.format(cp_type),
                                          'CUDA_memcpy_{}'.format(cp_type))
    line_cuda_mem_tab = line_cuda_mem.split(' ')
    line_cuda_mem_tab = [i for i in line_cuda_mem_tab if i]
    line_cuda_mem_tab[-1] = line_cuda_mem_tab[-1].replace(
        '[', '').replace(']', '')
    # clean tab
    line_cuda_mem_tab[0] = float(
        line_cuda_mem_tab[0].replace('%', ''))  # Time(%)
    line_cuda_mem_tab[1] = to_milli_second(line_cuda_mem_tab[1])  # Time (ms)
    line_cuda_mem_tab[2] = int(line_cuda_mem_tab[2])  # Calls
    line_cuda_mem_tab[3] = to_milli_second(line_cuda_mem_tab[3])
    line_cuda_mem_tab[4] = to_milli_second(line_cuda_mem_tab[4])
    line_cuda_mem_tab[5] = to_milli_second(line_cuda_mem_tab[5])
    return line_cuda_mem_tab


def extract_memcopy_cuda(nvprof_out_path):
    cuda_memcpy_list = []
    with open(nvprof_out_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            if re.search('CUDA memcpy HtoD', line) is not None:
                cuda_memcpy_list += [get_mem_copy(line, 'HtoD')]
            if re.search('CUDA memcpy DtoD', line) is not None:
                cuda_memcpy_list += [get_mem_copy(line, 'DtoD')]
            if re.search('CUDA memcpy DtoH', line) is not None:
                cuda_memcpy_list += [get_mem_copy(line, 'DtoH')]
    return cuda_memcpy_list


def save_in_csv(memcpy_name, info, nb_gpus, gbs, model):
    filename_csv = "{}.csv".format(memcpy_name)
    new_file = osp.isfile(filename_csv)
    with open(filename_csv, 'a') as f:
        if not new_file:
            f.write("NB_GPUs,GBS,Model,Time(%),Time(ms),Calls,Avg(ms),Min(ms),Max(ms)\n")
        f.write("{},{},{},{},{},{},{},{},{}\n".format(nb_gpus,
                                                   gbs,
                                                   model,
                                                   round(info['Time(%)'], 3),
                                                   round(info['Time(ms)'], 3),
                                                   info['Calls'],
                                                   round(info['Avg(ms)'], 3),
                                                   round(info['Min(ms)'], 3),
                                                   round(info['Max(ms)'], 3)))


def cuda_memcpy_to_csv(nb_gpus, gbs, model):
    list_nvprof_out = glob('nvprof_out*')
    list_mem = []
    for nvprof_out in list_nvprof_out:
        list_mem += extract_memcopy_cuda(nvprof_out)
    print(list_mem)
    df = pd.DataFrame(list_mem, columns=[
                      "Time(%)", "Time(ms)", "Calls", "Avg(ms)", "Min(ms)", "Max(ms)", "Name"])
    groups = df.groupby('Name')
    for name, df_group in groups:
        print(name)
        print(df_group.mean(numeric_only=True))
        save_in_csv(name, df_group.mean(numeric_only=True), nb_gpus, gbs, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_gpus',
                        type=int,
                        help='nb_gpus')
    parser.add_argument('--gbs',
                        type=int,
                        help='Global Batch Size')
    parser.add_argument('--model',
                        type=str,
                        help='Model Name')
    args = parser.parse_args()
    cuda_memcpy_to_csv(args.nb_gpus, args.gbs, args.model)
