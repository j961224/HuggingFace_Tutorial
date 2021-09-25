# End-of-chapter quiz

## 1. roberta-large-mnli checkpoint는 어떤 task를 수행할 수 있는가?

roberta는 Encoder를 주로 사용하고 문장 분류, NER, QnA에 사용되므로 Text classification이 정답이다!

## 2. code에서 어떤 것이 return 되는가?

~~~
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
~~~

NER로 개체명 인식하고 grouped_entities=True로 group화하여 개채명(persons, organizations 등)을 정해준다!


## 3. ... 부분을 대체할만한 것은?

~~~
from transformers import pipeline

filler = pipeline("fill-mask", model="bert-base-cased")
result = filler("...")
~~~

fill-mask이므로 [MASK]가 포함된 문장과 같은 것이 등장해야한다.

## 4. code 실패 이유는?

~~~
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier("This is a course about the Transformers library")
~~~

분류할만한 label을 주지 않아서 오류가 난다! (classifier("This ~",candidate_labels=[~]) 식으로 줘야한다.)


## 5. transfer learning 의미?

전이 학습은 사전 학습 모델을 가져와 task에 맞게 fine-tuning 하는 것이다.

## 6. Pretraining은 labels을 필요로 하지 않는다.

pretraining은 self-supervised이므로 자동적으로 알아서 labels을 만들어서 학습한다.

## 7. model, architecture, weights 관점에서 sentence 고르기

architecture -> 수학적 function의 set

weights -> 다른 parameters

## 8. text 생성과 관련 있는 model type은?

A decoder model

## 9. 요약 task와 관련 있는 model type은?

Seq2Seq model

## 10. text 분류와 관련 있는 model type은?

A encoder model

