# import openai
#
# api_key = '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'
# openai.api_key = api_key
# client = openai.Client(api_key)
#
# for i in range(5):
#     client.chat.completion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": "What is the capital of the United States?"},
#         ],
#         max_tokens=10,
#     )

# import numpy as np
# array1 = np.array([1, 2, 3, 4, 5])
# array2 = array1[(array1 > 2) & (array1 < 5)]
# print(array2)

import collections
test = collections.Counter()
test['a'] += 1
test['a'] += 1
test['b'] += 1
print(test)