import json, pickle
if __name__=='__main__':
    ann_path = '/home/u6693411/amodal_dataset/d2sa/D2S_amodal_validation.json'

    with open(ann_path, "r") as f:
        ann_data = json.load(f)

    cat_list = []
    for ann_cat in ann_data['categories']:
        cat_list.append(ann_cat['name'])

    print(cat_list)
    with open('/work/u6693411/aisformer/data/datasets/D2SA/d2sa_cat_list', 'wb') as fp:
        pickle.dump(cat_list, fp)

