# 最终源码真实集成：执行协议

输出使用新目录 `reports/revision/integration-closure`，不得覆盖此前运行。固定回答器不调用真实模型；复用工作区已有 BGE-M3、FAISS 索引、Cross-Encoder、SQLite 和 Chroma。Redis TTL 设1秒，验证6轮提交、缓存过期恢复、同 turn_id 重放、部分索引删除与修复，关闭外部遥测和问题拆分。脚本在执行前后分别计算全部运行源码 SHA-256，不一致时不得判通过；结果记录运行配置、库版本、Redis服务版本和语料哈希。

Redis 仅使用仓库 `tmp/memory-redis/extracted/usr/bin/redis-server` 及其已存在的库；启动前确认 Windows 和 Ubuntu 的 16488 端口无监听、`tmp/closure-redis-16488.pid` 不存在。新建隔离实例只绑定 127.0.0.1:16488，关闭 RDB/AOF，PID 文件写工作区 tmp。执行后核对本次 PID/端口与进程命令行，仅关闭该实例；不重启宿主、不停止其他服务或修改旧数据。
