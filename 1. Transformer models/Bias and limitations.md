# Bias and limitations

사전학습모델이나 fine-tuning을 이용하면 강력하지만 주의할 점이 있다!

**가장 큰 것은 많은 양의 데이터를 이용하니 안 좋은 데이터도 있을 수 있다.**

예시로 BERT 모델에 fill-mask를 들어보자!

~~~
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("This man works as a [MASK].")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print([r["token_str"] for r in result])

['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']
['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
~~~

이렇게 MASK된 부분을 채우는데 상위 5개 중에, prostitute가 나왔다.

-> 이러한 경우는 BERT는 BookCorpus나 위키피디아와 같은 데이터로 학습해서 드문 경우이다.

---

이러한 경우는 fine-tuning 만으로는 bias가 사라지지는 않는다!
