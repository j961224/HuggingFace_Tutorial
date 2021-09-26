# Behind the pipeline

## Preprocessing with a tokenizer

raw text -> convert the text inputs into numbers

* 입력을 word, subword or symbols token으로 분할한다.
* token -> integer로 mapping
* 추가로 입력 추가!

---

모든 전처리는 AutoTokenizer class와 from_pretrained metohd를 사용해 자동으로 cache한다!

~~~
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
~~~

-> tokenizer로 직접 문장을 처리할 수 있고 dictionary로 다시 돌아갈 수 있다!

* Transformer model은 Tensor로 입력을 받는다! -> Tensor를 NumPy 배열이라고 생각해도 됨!

-> tensor의 유형을 지정할 시, return_tensors 인수 사용!

~~~
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.", 
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
~~~

~~~
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
~~~

* inputs_ids: 각 문장의 토큰의 고유 정수로 mapping한 것!

## Going through the model

* AutoModel class 사용하기

~~~
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
~~~

-> input을 받아 hidden states를 output으로 낸다!

## A high-dimensional vector?

일반적으로 3개의 차원을 보여준다!

* Batch size: sequence 갯수
* Sequence length: sequence의 표현 길이
* Hidden size: 각 모델의 vector 차원 수

~~~
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

torch.Size([2, 16, 768])
~~~

## Model heads: Making sense out of numbers

![tt](https://user-images.githubusercontent.com/59636424/134796139-324b06aa-bc88-41e1-b62f-6aa1e74e3ece.PNG)

embeddings layer + subsequent layers로 이뤄짐!

* embeddings layer: 토큰화된 input ID를 vector로 변환!
* subsequent layer: attention을 이용해 vector를 조작해 최종 representation을 생성!

* AutoModel class를 사용하지 않고 AutoModelForSequenceClassification 사용!

~~~
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)

torch.Size([2, 2])
~~~

-> 2 x 2 인 이유: two sentences x two labels

## Postprocessing the output

~~~
print(outputs.logits)

tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
~~~

-> softmax 통과 전, logits 값이다!

~~~
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
~~~

-> softmax를 통한 확률 점수를 얻을 수 있다!

~~~
model.config.id2label

{0: 'NEGATIVE', 1: 'POSITIVE'}
~~~

-> model.config.id2label을 통해 id의 label 이름을 알 수 있다!






