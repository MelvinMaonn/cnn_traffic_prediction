# DCNN

#### 项目介绍
this is a deep CNN(DCNN) model for traffic prediction.


#### 软件架构
    read_data: this is a data generator to feed your model
    utils: this is a settings file for the model params
    config: this is also a settings file for some varibles
    main: this is the train code


#### 安装教程

1. the tensorflow version is 3.x
2. you should create some dirs, which would be used:
    checkpoint/
    logdir/


#### 使用说明

1. you should update the path of your data
2. if you have gpu you can use it by setting like 'os.environ['CUDA_VISIBLE_DEVICES'] = '4'' in main.py
3. there are train and val code in the train(), though it complex, it still works, and you can refactor it