
import xmltodict, collections, cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_dict(path):
    x = xmltodict.parse(open(path, 'r').read())
    task_name = x['annotations']['meta']['task']['name']
    video_name = 'none'

    tracks = []
    if not 'track' in x['annotations']:
        return
    if isinstance(x['annotations']['track'], collections.OrderedDict):
        track_list = [x['annotations']['track']]
    elif isinstance(x['annotations']['track'], list):
        track_list = x['annotations']['track']

    for i in track_list:
        label = i['@label']

        frames = {}
        if 'polygon' in i.keys():
            if isinstance(i['polygon'], collections.OrderedDict):
                poly_list = [i['polygon']]
            elif isinstance(i['polygon'], list):
                poly_list = i['polygon']
            for j in poly_list:
                frame = j['@frame']
                outside = int(j['@outside'])
                if outside==1:
                    continue
                points = np.array([
                                 [q.split(',') for q in j['@points'].split(';')]
                ], 'float32')
                frames[frame] = {'points': points, 'subclass': 'none'}
            tracks.append([{'label':label, 'frames':frames, 'type': 'polygon'}])

        elif 'points' in i.keys():
            if isinstance(i['points'], collections.OrderedDict):
                points_list = [i['points']]
            elif isinstance(i['points'], list):
                points_list = i['points']
            for j in points_list:
                frame = j['@frame']
                outside = int(j['@outside'])
                if outside==1:
                    continue
                points = np.array([
                                 [q.split(',') for q in j['@points'].split(';')]
                ], 'float32')
                point_id = j['attribute']['#text'] if 'attribute' in j.keys() else 'none'
                frames[frame] = {'points': points, 'subclass': point_id}
            tracks.append([{'label':label, 'frames':frames, 'type': 'points'}])

        elif 'polyline' in i.keys():
            if isinstance(i['polyline'], collections.OrderedDict):
                points_list = [i['polyline']]
            elif isinstance(i['polyline'], list):
                points_list = i['polyline']
            for j in points_list:
                frame = j['@frame']
                outside = int(j['@outside'])
                if outside==1:
                    continue
                points = np.array([
                                 [q.split(',') for q in j['@points'].split(';')]
                ], 'float32')
                frames[frame] = {'points': points, 'subclass': 'none'}
            tracks.append([{'label':label, 'frames':frames, 'type': 'polyline'}])

        else:
            pass

    return {
      'task_name':task_name,
      'video_name':video_name,
      'tracks':tracks
    }


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
      #  raise Exception('lines do not intersect')
      return 0,0

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def create_line(p1, p2):
    """
    Получаем прямую, проходящую через 2 точки
    params: two tuples (int)
    return: координаты прямой
    """
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, C

def calc_line(p1,p2, x1, x2):
    A,B,C = create_line(p1, p2)
def f(x):
    return (-C - A*x) / B
    return f(x1), f(x2)

def get_dots_for_verticals_and_sectors(verticals, sectors):
    dots=[]
    for v in verticals:
        for i in range(len(v)-1):
            line = [v[i], v[i+1]]
            for s in sectors:
                x,y = line_intersection(line, s)
                if x<=max(line[0][0],line[1][0]) and x>=min(line[0][0],line[1][0]) \
                      and y<=max(line[0][1],line[1][1]) and y>=min(line[0][1],line[1][1]) or \
                      abs(x-max(line[0][0],line[1][0])) < 5 and abs(y-max(line[0][1],line[1][1])) < 5 or \
                       abs(x-min(line[0][0],line[1][0])) < 5 and abs(y-min(line[0][1],line[1][1])) < 5 :
                    dots.append([x,y])
    return dots_delete_duples(dots)

def dots_delete_duples(dots):
    newj = []
    for k in dots:
        if k[0]==0 and k[1]==0:
            isin=True
            break
        isin=False
        for k2 in newj:
            dist = np.sqrt((k[0]-k2[0])**2 + (k[1]-k2[1])**2)
            if dist<3:
                isin = True
                break
        if not isin:
            newj.append(k)
    return newj