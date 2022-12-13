from django.http import JsonResponse, QueryDict
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

'''
아이리스 품종 3: setosa, versicolor, virginica
'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'
꽃받침 길이, 꽃받침 너비, 꽃잎 길이, 꽃잎 너비

QueryDict 관련 블로그
https://velog.io/@qlgks1/Django-request-HttpRequest-QueryDict
'''
@api_view(['POST'])
@parser_classes([JSONParser])
def find_iris(request):
    print(f'리액트에서 보낸 데이터: {request.data}')
    sepal_length_cm = QueryDict.getitem("SepalLengthCm")
    print(f'넘어온 꽃받침 길이: {sepal_length_cm}')
    print(f'찾는 품종: {request.data}')
    return JsonResponse({'result':'virginica'})