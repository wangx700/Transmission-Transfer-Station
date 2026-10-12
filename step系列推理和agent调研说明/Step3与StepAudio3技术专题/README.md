# Step-3与StepAudio 3：推理与Agent技术专题

**资料时点：2026-10-12，北京时间。** 依据论文正文及已读取的公开接口整理。论文结果不是本次复现；流程中的工程扩展均明确标注，不代表官方生产实现或昇腾已适配。

## 1. 先讲什么：按推理与Agent的相关性排序

表注：优先级是面向本次推理/Agent技术交流的编辑建议，不是论文或模型排名。日期为用户给出的发表月份；三篇Audio 3属于不同任务方向，不拼成一个模型架构。

| 优先级 | 论文 | 推荐主标题 | 最适合的听众 |
|---|---|---|---|
| P0 | Step-3，2025-07 | 激活参数更少，为何不一定解码更便宜？ | 推理内核、分布式服务、昇腾性能团队 |
| P0 | StepAudio 3 Realtime，2026-09 | 边听、边说、边思考、边执行的语音Agent | Agent框架、实时服务与后训练团队 |
| P1 | StepAudio 3 Gen，2026-09 | 将时间预测与帧内声学细节分开计算 | 音频推理、codec与统一生成团队 |
| P1 | StepAudio 3 Music，2026-09 | 先生成可编辑计划，再生成与渲染音乐 | 内容生成Agent、长序列与多阶段推理团队 |

介绍不必四篇平均分配篇幅。推理场合先讲Step-3和Gen；Agent场合先讲Realtime和Music中的计划编辑接口。

## 文档导航

- [01-Step3解码成本与模型系统协同](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/Step3%E4%B8%8EStepAudio3%E6%8A%80%E6%9C%AF%E4%B8%93%E9%A2%98/01-Step3%E8%A7%A3%E7%A0%81%E6%88%90%E6%9C%AC%E4%B8%8E%E6%A8%A1%E5%9E%8B%E7%B3%BB%E7%BB%9F%E5%8D%8F%E5%90%8C.md)
- [02-StepAudio3Realtime推理与语音Agent](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/Step3%E4%B8%8EStepAudio3%E6%8A%80%E6%9C%AF%E4%B8%93%E9%A2%98/02-StepAudio3Realtime%E6%8E%A8%E7%90%86%E4%B8%8E%E8%AF%AD%E9%9F%B3Agent.md)
- [03-StepAudio3Gen时间深度建模与生成推理](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/Step3%E4%B8%8EStepAudio3%E6%8A%80%E6%9C%AF%E4%B8%93%E9%A2%98/03-StepAudio3Gen%E6%97%B6%E9%97%B4%E6%B7%B1%E5%BA%A6%E5%BB%BA%E6%A8%A1%E4%B8%8E%E7%94%9F%E6%88%90%E6%8E%A8%E7%90%86.md)
- [04-StepAudio3Music规划与多阶段推理](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/Step3%E4%B8%8EStepAudio3%E6%8A%80%E6%9C%AF%E4%B8%93%E9%A2%98/04-StepAudio3Music%E8%A7%84%E5%88%92%E4%B8%8E%E5%A4%9A%E9%98%B6%E6%AE%B5%E6%8E%A8%E7%90%86.md)
- [05-昇腾适配边界与配图说明](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/Step3%E4%B8%8EStepAudio3%E6%8A%80%E6%9C%AF%E4%B8%93%E9%A2%98/05-%E6%98%87%E8%85%BE%E9%80%82%E9%85%8D%E8%BE%B9%E7%95%8C%E4%B8%8E%E9%85%8D%E5%9B%BE%E8%AF%B4%E6%98%8E.md)

## 范围说明

优先展开Step-3的解码成本与AFD、Realtime的推理/播放/工具并发；Gen和Music分别讲生成计算分工与可编辑计划。模型代际独立，不能把MFA、Sparse GQA和Audio3生成路径合并成Step5内部实现。正式实现以公开代码为准；未取得代码证据的细节保持论文级说明。
