"""
TikTok Video Classification Script

This script processes a TikTok video URL and classifies the content as either "claims" or "opinions" 
with the model that is developed during this project.
Using the pyktok package: https://github.com/dfreelon/pyktok to extract the metadata of TikTok videos.
"""

import pandas as pd
import joblib, os, io, contextlib, sys


def suppress_prints():
    return contextlib.redirect_stdout(io.StringIO())



def load_data_and_model(url):

    # Suppress print statements during the import
    with suppress_prints():
        import pyktok as pyk

        path = 'video_data.csv'
        if os.path.exists(path):
            os.remove(path)


        pyk.save_tiktok(url, False, path, 'chrome')


    data = pd.read_csv(path)
    download_model = joblib.load("docs/datasets/download_predictor.joblib")
    model = joblib.load("docs/datasets/best_model.joblib") # main classifier

    return data, download_model, model

def preprocess(data, download_model):
    data["author_ban_status"] = 1.0 # assumption: the author of the video that you saw is not banned

    filter_without_download = ['author_verified',
              'author_ban_status',
              'video_playcount',
              'video_diggcount',
              'video_sharecount',
              'video_commentcount'
            ]

    name_remapping = dict(zip(filter_without_download,download_model.feature_names_in_))

    filtered_data = data[filter_without_download]
    filtered_data = filtered_data.rename(columns=name_remapping)

    filtered_data["video_download_count"] = download_model.predict(filtered_data) 
    filtered_data["verified_status"] = filtered_data["verified_status"].astype(float)

    # Reorder the columns like the training dataset for the model
    reordering = ['verified_status', 'author_ban_status', 'video_view_count',
           'video_like_count', 'video_share_count', 'video_download_count', 'video_comment_count']
    X = filtered_data[reordering]

    return X
    

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        print(" Url is missing. Example: python3 script.py https://www.tiktok.com/@paralympics/video/7375186925335924001")
        sys.exit()
    
    data, download_model, model = load_data_and_model(url)
    X = preprocess(data, download_model)

    classification = model.predict(X)

    result = "claims" if classification else "opinions"

    print(f"The TikTok video with the url {url} contains {result}")
    


if __name__ ==  "__main__":
    main()