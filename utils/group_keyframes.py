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