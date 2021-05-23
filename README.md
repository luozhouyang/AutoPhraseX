# AutoPhraseX

![Python package](https://github.com/luozhouyang/autophrasex/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/autophrasex.svg)](https://badge.fury.io/py/autophrasex)
[![Python](https://img.shields.io/pypi/pyversions/autophrasex.svg?style=plastic)](https://badge.fury.io/py/autophrasex)


Automated Phrase Mining from Massive Text Corpora in Python.


实现思路参考 [shangjingbo1226/AutoPhrase](https://github.com/shangjingbo1226/AutoPhrase)，并不完全一致。

## 安装

```bash
pip install -U autophrasex
```

## 基本使用

```python
from autophrasex import *

# 构造autophrase
autophrase = AutoPhrase(
    reader=DefaultCorpusReader(tokenizer=JiebaTokenizer()),
    selector=DefaultPhraseSelector(),
    extractors=[
        NgramsExtractor(N=4), 
        IDFExtractor(), 
        EntropyExtractor()
    ]
)

# 开始挖掘
predictions = autophrase.mine(
    corpus_files=['data/answers.10000.txt'],
    quality_phrase_files='data/wiki_quality.txt',
    callbacks=[
        LoggingCallback(),
        ConstantThresholdScheduler(),
        EarlyStopping(patience=2, min_delta=3)
    ])

# 输出挖掘结果
for pred in predictions:
    print(pred)

```


## 高级用法

本项目的各个关键步骤都是可以扩展的，所以大家可以自由实现自己的逻辑。

本项目大体上可以氛围以下几个主要模块：

* `tokenizer`分词器模块
* `reader`语料读取模块
* `selector`高频短语的选择模块
* `extractors`特征抽取器，用于抽取分类器所需要的特征
* `callbacks`挖掘周期的回调模块

以下是每个模块的高级使用方法。

### tokenizer

`tokenizer`用于文本分词，用户可以继承`AbstractTokenizer`实现自己的分词器。本库自带`JiebaTokenizer`。

例如，你可以使用`baidu/LAC`来进行中文分词。你可以这样实现分词器：

```python
# pip install lac

class BaiduLacTokenizer(AbstractTokenizer):

    def __init__(self, custom_vocab_path=None, model_path=None, mode='seg', use_cuda=False, **kwargs):
        self.lac = LAC(model_path=model_path, mode=mode, use_cuda=use_cuda)
        if custom_vocab_path:
            self.lac.load_customization(custom_vocab_path)

    def tokenize(self, text, **kwargs):
        text = self._uniform_text(text, **kwargs)
        results = self.lac.run(text)
        results = [x.strip() for x in results if x.strip()]
        return results
```

然后在构建`reader`的使用使用`BaiduLacTokenizer`:
```python
reader = DefaultCorpusReader(tokenizer=BaiduLacTokenizer())
```

### reader

`reader`用于读取语料，用户可以继承`AbstractCorpusReader`实现自己的Reader。本库自带`DefaultCorpusReader`。

因为目前的`extractor`其实是依赖`reader`的（具体来说是`extractor`实现了`reader`的生命周期回调接口），因此想要重写`reader`，在有些情况下需要同时更改`extractor`的实现，此时自定义成本比较大，暂时不推荐重写`reader`。

### selector

`selector`用于选择高频Phrase，用户可以继承`AbstractPhraseSelector`实现自己的Phrase选择器。本库自带`DefaultPhraseSelector`。

`selector`可以拥有多个`phrase_filter`，用于实现Phrase的过滤。关于`phrase_filter`本库提供了开放的接口，用户可以继承`AbstractPhraseFilter`实现自己的过滤器。本库自带了默认的过滤器`DefaultPhraseFilter`，并且在默认情况下使用。

如果你想要禁用默认的过滤器，转而使用自己实现的过滤器，可以在构造`selector`的时候设置：

```python
# 自定义过滤器
class MyPhraseFilter(AbstractPhraseFilter):

    def apply(self, pair, **kwargs):
        phrase, freq = pair
        # return True to filter this phrase
        if is_verb(phrase):
            return True
        return False

selector = DefaultPhraseSelector(
    phrase_filters=[MyPhraseFilter()], 
    use_default_phrase_filters=False
)
```

考虑到有些过滤过程，使用按批处理可以显著提升速度(例如使用深度学习模型计算词性)，`phrase_filter`提供了一个`batch_apply`方法。

举个例子，使用`baidu/LAC`来计算词性，从而实现Phrase的过滤：

```python

class VerbPhraseFilter(AbstractPhraseFilter):

    def __init__(self, batch_size=100):
        super().__init__()
        self.lac = LAC()
        self.batch_size = batch_size

    def batch_apply(self, batch_pairs, **kwargs):
        predictions = []
        for i in range(0, len(batch_pairs), self.batch_size):
            batch_texts = [x[0] for x in batch_pairs[i: i + self.batch_size]]
            batch_preds = self.lac.run(batch_texts)
            predictions.extend(batch_preds)
        candidates = []
        for i in range(len(predictions)):
            _, pos_tags = predictions[i]
            if any(pos in ['v', 'vn', 'vd'] for pos in pos_tags):
                continue
            candidates.append(batch_pairs[i])
        return candidates

selector = DefaultPhraseSelector(
    phrase_filters=[VerbPhraseFilter()], 
    use_default_phrase_filters=False
)
```

### extractor

`extractor`用于抽取分类器的特征。特征抽取器会在`reader`读取语料的时候进行必要信息的统计。因此`extractor`实现了`reader`的回调接口，所以在自定义特征抽取器的时候，需要同时继承`AbstractCorpusReadCallback`和`AbstractFeatureExtractor`。

本库自带了以下几个特征抽取器：

* `NgramExtractor`，`n-gram`特征抽取器，可以计算phrase的`pmi`特征
* `IDFExtractor`，`idf`特征抽取器，可以计算phrase的`doc_freq`、`idf`特征
* `EntropyExtractor`，`熵`特征抽取器，可以计算phrase的`左右熵`特征

上述自带的特征抽取器，都是基于`n-gram`统计的，因此都支持`ngram`的选择，也就是都可以自定义`ngram_filter`来过滤不需要统计的`ngram`。本库自带了`DefaultNgramFilter`，并且默认启用。用户可以实现自己的`ngram_filter`来灵活选取合适的`ngram`。

举个例子，我需要过滤掉`包含标点符号`的`ngram`：

```python
CHARACTERS = set('!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ \t\n\r\x0b\x0c，。？：“”【】「」')

class MyNgramFilter(AbstractNgramFiter):

    def apply(self, ngram, **kwargs):
        if any(x in CHARACTERS for x in ngram):
            return True
        return False

autophrase = AutoPhrase(
    reader=DefaultCorpusReader(tokenizer=JiebaTokenizer()),
    selector=DefaultPhraseSelector(),
    extractors=[
        NgramsExtractor(N=4, ngram_filters=[MyNgramFilter()]), 
        IDFExtractor(ngram_filters=[MyNgramFilter()]), 
        EntropyExtractor(ngram_filters=[MyNgramFilter()]),
    ]
)
# 开始挖掘
...
```

用户可以继承`AbstractFeatureExtractor`实现自己的特征计算。只需要在构建autophrase实例的时候，把这些特征计算器传入即可，不需要做其他任何额外操作。

举个例子，我增加一个`phrase是否是unigram`的特征：

```python
class UnigramFeatureExtractor(AbstractFeatureExtractor，AbstractCorpusReadCallback):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def extract(self, phrase, **kwargs):
        parts = phrase.split(' ')
        features = {
            'is_unigram': 1 if len(parts) == 1 else 0
        }
        return features


autophrase = AutoPhrase(
    reader=DefaultCorpusReader(tokenizer=JiebaTokenizer()),
    selector=DefaultPhraseSelector(),
    extractors=[
        NgramsExtractor(N=N), 
        IDFExtractor(), 
        EntropyExtractor(),
        UnigramFeatureExtractor(),
    ]
)

# 可以开始挖掘了
...
```

### callback

`callback`回调接口，可以提供phrase挖掘过程中的生命周期监听，并且实现一些稍微复杂的功能，例如`EarlyStopping`、`判断阈值Schedule`等。

本库自带以下回调：

* `LoggingCallback`提供挖掘过程的日志信息打印
* `ConstantThresholdScheduler`在训练过程中调整阈值的回调
* `EarlyStopping`早停，在指标没有改善的情况下停止训练

用户可以自己继承`Callback`实现自己的逻辑。


## 结果示例

> 以下结果属于本库比较早期的测试效果，目前本库的代码更新比较大，返回结果和下述内容不太一致。仅供参考。

新闻语料上的抽取结果示例：

```bash
成品油价格, 0.992766816097071
股份制银行, 0.992766816097071
公务船, 0.992766816097071
中国留学生, 0.992766816097071
贷款基准, 0.992766816097071
欧足联, 0.992766816097071
新局面, 0.992766816097071
淘汰赛, 0.992766816097071
反动派, 0.992766816097071
生命危险, 0.992766816097071
新台阶, 0.992766816097071
知名度, 0.992766816097071
新兴产业, 0.9925660976153782
安全感, 0.9925660976153782
战斗力, 0.9925660976153782
战略性, 0.9925660976153782
私家车, 0.9925660976153782
环球网, 0.9925660976153782
副校长, 0.9925660976153782
流行语, 0.9925660976153782
债务危机, 0.9925660976153782
保险资产, 0.9920376397372204
保险机构, 0.9920376397372204
豪华车, 0.9920376397372204
环境质量, 0.9920376397372204
瑞典队, 0.9919345469537152
交强险, 0.9919345469537152
马卡报, 0.9919345469537152
生产力, 0.9911077251879798
```

医疗对话语料的抽取示例：

```bash
左眉弓, 1.0
支原体, 1.0
mri, 1.0
颈动脉, 0.9854149008885851
结核病, 0.9670815675552518
手术室, 0.9617546444783288
平扫示, 0.9570324222561065
左手拇指, 0.94
双膝关节, 0.94
右手中指, 0.94
拇指末节, 0.94
cm皮肤, 0.94
肝胆脾, 0.94
抗体阳性, 0.94
igm抗体阳性, 0.94
左侧面颊, 0.94
膀胱结石, 0.94
左侧基底节, 0.94
腰椎正侧, 0.94
软组织肿胀, 0.94
手术瘢痕, 0.94
枕顶部, 0.94
左膝关节正侧, 0.94
膝关节正侧位, 0.94
腰椎椎体, 0.94
承德市医院, 0.94
性脑梗塞, 0.94
颈椎dr, 0.94
泌尿系超声, 0.94
双侧阴囊, 0.94
右颞部, 0.94
肺炎支原体, 0.94
```