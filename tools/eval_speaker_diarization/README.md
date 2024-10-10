# 说话人日志效果评估

1. 安装依赖库

```shell
pip install pyannote.audio[separation]==3.3.0
```

2. 下载[AIShell-4](https://us.openslr.org/resources/111)的测试数据并解压到当前目录的`dataset`下。
3. 执行`create_aishell4_test_rttm.py`，创建数据类别和rttm文件。
4. 执行`infer_data.py`预测数据。
5. 执行`compute_metrics.py`获取评估结果。
