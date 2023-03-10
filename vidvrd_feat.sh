for d in /DATA/kumar204/dataset/vidvrd/image/*; do
#  echo $d
    folder_name=`echo $d | cut -d "/" -f 7`
    echo $folder_name
    mkdir /DATA/kumar204/dataset/vidvrd/feature/$folder_name
    python /DATA/kumar204/feature_extraction/fatser_rcnn/vqa-maskrcnn-benchmark/extract_frcnn_feats.py --image_dir=$d --output_folder=/DATA/kumar204/dataset/vidvrd/feature/$folder_name 
    scp -r /DATA/kumar204/dataset/vidvrd/feature/$folder_name yogesh@10.6.0.34:/data1/yogesh/dataset/vidvrd/frcnn_feat/
    rm -r /DATA/kumar204/dataset/vidvrd/feature/$folder_name 
done
