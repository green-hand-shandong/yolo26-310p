#!/usr/bin/env python3
# 尝试 OBB OM 中逐个禁用节点的方式太复杂; 先确认是哪个算子
# 检查 Sin/Cos 是否有专门 TBE 算子失败 —— dump 全部节点
import re, subprocess
# 用 python 直接读 om 中的算子序列没有公开 API, 改从 CANN 日志开关入手
print("log approach: re-execute with acl.detail log enabled")
