import numpy as np
import json
import os.path
from copy import deepcopy
import sys 
from utils import load_config_file

DATA_CONFIG_PATH = "/data1/VRG/VRC/VR_SimilarityNetwork/VRG_configs/data_config_test.yaml"
TESTER_CONFIG_PATH = "/data1/VRG/VRC/VR_SimilarityNetwork/VRG_configs/test_config.yaml"

data_config = load_config_file(DATA_CONFIG_PATH)
test_config = load_config_file(TESTER_CONFIG_PATH)

# same function as above but uses starting and ending frame from ground truth vidvrd annotation file
def tracklet_gt(vid_name, pred, f_s, f_e):
    if os.path.isfile('/data1/VRG/STGCN/VideoVRD/preprocess_data/tracking/videovrd_detect_tracking/{}.npy'.format(vid_name)):
        cur_gt_stgcn = np.load('/data1/VRG/STGCN/VideoVRD/preprocess_data/tracking/videovrd_detect_tracking/{}.npy'.format(vid_name))
    if os.path.isfile('/data1/VRG/STGCN/VideoVRD/vidvrd-dataset/test/{}.json'.format(vid_name)):
        cur_gt = json.load(open('/data1/VRG/STGCN/VideoVRD/vidvrd-dataset/test/{}.json'.format(vid_name)))
    elif os.path.isfile('/data1/VRG/STGCN/VideoVRD/vidvrd-dataset/train/{}.json'.format(vid_name)):
        cur_gt = json.load(open('/data1/VRG/STGCN/VideoVRD/vidvrd-dataset/train/{}.json'.format(vid_name)))
            
    else:
        print("file not found")
    
    final_traj = {}
    sub = pred.split('-')[0]
    obj = pred.split('-')[2]

    sub_tid = []
    obj_tid = []
    obj_b = {}
    sub_b = {}


    for ele in cur_gt['subject/objects']:
        if ele['category'] == sub:
            sub_tid.append(ele['tid'])
        if ele['category'] == obj:
            obj_tid.append(ele['tid'])

    for ele in cur_gt['relation_instances']:
        #print(ele['predicate'])
        if ele['predicate'] == pred.split('-')[1] and (ele['subject_tid'] in sub_tid) and (ele['object_tid'] in obj_tid):
            f_s = ele['begin_fid']
            f_e = ele['end_fid']
            print(f'relation {pred} found in video {vid_name} with duration {f_s} to {f_e}')
        


            for ele in cur_gt_stgcn:
                if int(ele[1]) in sub_tid:
                    #print(f'relation {pred} found in video {vid_name} with duration {f_s} to {f_e}')
                    for ele1 in cur_gt_stgcn:
                        if int(ele1[1]) == int(ele[1]) and int(ele1[0] ) >= f_s and int(ele1[0]) < f_e:
                            x1, y1, x2, y2 = ele1[2:6]
                            sub_b[str(int(ele1[0]))] = [x1, y1, x2 + x1 , y2 + y1]  #considering bboxes as: xmin, ymn, width, hight
                            #sub_b[str(int(ele1[0]))] = [x1, y1, x2 , y2] 
                    break
            
            for ele in cur_gt_stgcn:
                if int(ele[1]) in obj_tid:
                    for ele1 in cur_gt_stgcn:
                        if int(ele1[1]) == int(ele[1]) and int(ele1[0] - 1) >= f_s and int(ele1[0] - 1) < f_e:
                            x1, y1, x2, y2 = ele1[2:6]
                            obj_b[str(int(ele1[0]))] = [x1, y1, x2 +x1 ,y2 + y1]
                            #sub_b[str(int(ele1[0]))] = [x1, y1, x2 , y2] 
                    break
            break

                

    final_traj["sub"] = sub_b
    final_traj["obj"] = obj_b

    return final_traj

    # Extracting tracklets of all proposals from stgcn proposals
def extract_tracklets(gt_prs,id_to_obj):
    fs_eval_gt = dict.fromkeys(gt_prs['results'])
    for i, vid_name in enumerate(fs_eval_gt.keys()):
        dup_pred = []
        fs_eval_gt[vid_name] = {}
        for j, ele in enumerate(gt_prs['results'][vid_name]):
            if ele['triplet'][1][0] == "bg":
                #print("hi")
                continue
            # elif ele['triplet'][0] == ele['triplet'][2]:
            #     continue
            else:
                for pred in ele['triplet'][1]:
                    sub = id_to_obj[int(ele['triplet'][0])]
                    obj = id_to_obj[int(ele['triplet'][2])]
                    pred_t = ''.join([sub,'-',pred, '-', obj])
                    f_s = ele['duration'][0]
                    f_e = ele['duration'][1]
                    #print(f'Predicate: {pred_t} in video {vid_name}')
                    #
                    #print(pred_t)
                    # if pred not in confirm_pred:
                    #     confirm_pred.append(pred)
                    if pred_t in dup_pred:
                        continue
                    dup_pred.append(pred_t)
                    #print(f'Predicate: {pred_t} in video {vid_name}')
                    fs_eval_gt[vid_name][pred_t] = tracklet_gt(vid_name.split('-')[0], pred_t,f_s,f_e, )
    return fs_eval_gt


#Extracting stgcn proposal bboxes and vidvrd ground truth bboxes for evaluation:
#It take care of multiple relations in same video
def create_eval_file(idx,filename,gt_prs):
    out = json.load(open(data_config.result))
    info = []
    vid_name = {}
    avail = []
    pred_s = []
    pred = []
    v_names = []
    for i, pred_name in enumerate(out.keys()):
        f = 0
        for j, ele in enumerate(out[pred_name]):
            info.append([ele[idx]['video_name'], ele[idx]['proposal_idx'], ele[idx]['has_common_pred'], ele[idx]['common_pred']])
            key = ele[idx]['video_name'] + '-' + str(i)
            vid_name[key] = [ele[idx]['proposal_idx'], ele[idx]['common_pred'], ele[idx]['has_common_pred']]
            avail.append(ele[idx]['has_common_pred'])
            pred_s.append(ele[idx]['common_pred'])
            v_names.append(key)
            break



    out_final = deepcopy(gt_prs) # it will contain only desired videos and proposals for evaluation



    for vid in vid_name.keys():
            out_final["results"][vid] = gt_prs["results"][vid.split('-')[0]]
            gt_prs["results"][vid] = gt_prs["results"][vid.split('-')[0]]

    # out_final2 = deepcopy(out_final1)

    # for vid in gt_prs["results"].keys():
    #     if vid in out_final2["results"].keys():
    #         #print("yes")
    #         del out_final2["results"][vid]



    for vid in gt_prs["results"]:
        if vid in vid_name:
            idx = []
            for i, prop in enumerate(gt_prs["results"][vid]):
                if vid_name[vid][0] == i: 
                    if data_config.mode == 1 and vid_name[vid][2] == 1:
                        out_final["results"][vid][vid_name[vid][0]]['triplet'][1] = [vid_name[vid][1]] # copying and removing other predicate
                    elif data_config.mode == 0:
                        out_final["results"][vid][vid_name[vid][0]]['triplet'][1] = [vid_name[vid][1]] # copying and removing other predicate

                else:
                    idx.append(i)
            out_final["results"][vid] = [j for i, j in enumerate(out_final["results"][vid]) if i not in idx]

            continue
        else:
            del out_final["results"][vid]
    
    return info, out_final

def main():
    print("Extracting proposal bboxes:")
    id_to_obj = {}
    with open('/data1/VRG/STGCN/VideoVRD/preprocess_data/tracking/cateid_name.txt') as f:
        for line in f:
            (key, val) = line.split()
            id_to_obj[int(key)] = val
    
    gt_prs = json.load(open(data_config.PROPOSALS_PATH))
    _, out_final = create_eval_file(test_config.Q_INDEX, 'result', gt_prs)
    gt_out_stgcn = extract_tracklets(out_final,id_to_obj)
    #saving without removing background/empty videos
    with open('gt_out_stgcn.json', 'w') as f:
        json.dump(gt_out_stgcn,f)

if __name__ == "__main__":
    main()

