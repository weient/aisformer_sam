import json, pickle
if __name__=='__main__':
    ann_path = '/home/weientai18/ais/data/datasets/COCOA/annotations/test.json'

    with open(ann_path, "r") as f:
        ann_data = json.load(f)

    cat_list = []
    for ann_cat in ann_data['categories']:
        cat_list.append(ann_cat['name'])

    print(cat_list)
    with open('/home/weientai18/ais/data/datasets/COCOA/cocoa_cat_list', 'wb') as fp:
        pickle.dump(cat_list, fp)

