import sys
import pyktok as pyk
import pandas as pd
pyk.specify_browser('chrome') #browser specification may or may not be necessary depending on your local settings

if len(sys.argv) >= 1:
    url = sys.argv[0]
else:
    print("No URL was given. Example: \n python3 script.py https://www.tiktok.com/@paralympics/video/7375186925335924001")

#url = "https://www.tiktok.com/@paralympics/video/7375186925335924001"
path = 'video_data.csv'

pyk.save_tiktok(url, True, path, 'chrome')
metadata = pd.read_csv(path)

