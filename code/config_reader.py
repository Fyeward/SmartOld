from configobj import ConfigObj
import numpy as np


def config_reader():
    config = ConfigObj('config')

    params = config['params']
    model_id = params['modelID']
    model = config['models'][model_id]
    model['boxsize'] = int(model['boxsize'])
    model['stride'] = int(model['stride'])
    model['padValue'] = int(model['padValue'])
    #param['starting_range'] = float(params['starting_range'])
    #param['ending_range'] = float(params['ending_range'])
    params['octave'] = int(params['octave'])
    params['use_gpu'] = int(params['use_gpu'])
    params['starting_range'] = float(params['starting_range'])
    params['ending_range'] = float(params['ending_range'])
    params['scale_search'] = map(float, params['scale_search'])
    params['thre1'] = float(params['thre1'])
    params['thre2'] = float(params['thre2'])
    params['thre3'] = float(params['thre3'])
    params['mid_num'] = int(params['mid_num'])
    params['min_num'] = int(params['min_num'])
    params['crop_ratio'] = float(params['crop_ratio'])
    params['bbox_ratio'] = float(params['bbox_ratio'])
    params['GPUdeviceNumber'] = int(params['GPUdeviceNumber'])

    return params, model

if __name__ == "__main__":
    config_reader()
