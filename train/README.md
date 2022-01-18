# Train
在术语标注过程中，我们预先统计好标准术语在常见语料中的document frequency(df), 然后按照(1-df)的概率正常标注BI，df的概率标注为O，以此来避免高频术语被学习过度。

## Quick Start
注意需要先下载pretrained models.
```
1. cd pretrain
2. sh download_pubmedbert.sh  ## 需要一段时间
3. cd ../train && sh run.sh
```