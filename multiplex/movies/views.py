from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from multiplex.movies.services import DcGan


@api_view(['GET'])
@parser_classes([JSONParser])
def faces(request):
    DcGan().show_face()
    print(f'Enter Show Faces with {request}')
    return JsonResponse({'Response Test ': 'SUCCESS'})
