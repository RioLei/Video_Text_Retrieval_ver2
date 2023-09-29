import re
def extractKeyframes(input_string):
    pattern = r'/([^/]+)/([^/]+)/\d+\.jpg$'
    match = re.search(pattern, input_string)
    return match.group(2)

def rearrangeDict(dict):
    grouped_data = {}
    for name in dict:
        keyframe_dir = extractKeyframes(name['imgpath'])
        list_frame = name
        if keyframe_dir not in grouped_data:
            grouped_data[keyframe_dir] = {"keyframe_dir": keyframe_dir, "list_frame": []}
        grouped_data[keyframe_dir]["list_frame"].append(name)
    output_dict = list(grouped_data.values())
    return output_dict

def deleteFrames(ids, dict, text_out):
    for i in ids:
        for j in dict:
            if j['keyframe_dir'] == i:
                dict.remove(j)
    data_str = ', '.join(map(str, dict))
    with open(text_out, 'w') as file:
        file.write(dict)
    return()

# ids = ['L01_V001', 'L02_V001',...]
# dict = input_dict
# text_out = output txt file
