# Models

## Creating a Transformer

~~~
import transformers
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)
~~~

-> cofig로 hidden_size, num_hidden_layers 등을 볼 수 있다.

## Different loading methods

~~~
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!
~~~

-> 이 상태로 모델 사용은 가능하지만 **training 시켜아한다!**

-> 불필요한 노력을 하지 않으려면 **pretrain된 모델을 사용하면 된다!**

~~~
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
~~~

* 이러한 과정을 AutmModel class로 대체 가능!

-> 위 코드는 Bert-base-case를 사용해 모델 로드함! -> 이거슨 모델 checkpoint!

-> checkpoint의 모든 가중치 초기화!

-> 새로운 task에 대해서 fine-tuning 가능! -> pretrain된 weights로 더 좋은 결과를 얻을 수 있다!

## Saving methods

-> save_pretrained로 모델 save!

~~~
model.save_pretrained("directory_on_my_computer")
~~~

![xx](https://user-images.githubusercontent.com/59636424/134797809-99752593-96c5-4234-8d4e-f7d93052d390.PNG)

* config.json: checkpoint 시작된 곳, 마지막으로 저장되었던 Transformers 버전과 같인 metadata 저장
* pytorch_model.bin: model weights가 포함된 state dictionary이다.

## Using a Transformer model for inference

* Transformer models는 numbers만 처리 가능!

* model의 inputs으로 들어오기 전에 수행해야 할 작업!

1. sequences

~~~
sequences = [
  "Hello!",
  "Cool.",
  "Nice!"
]
~~~

2. vocabulary index로 변환 & tensor로 변환

~~~
encoded_sequences = [
  [ 101, 7592,  999,  102],
  [ 101, 4658, 1012,  102],
  [ 101, 3835,  999,  102]
]

model_inputs = torch.tensor(encoded_sequences)
~~~
