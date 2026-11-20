"""Render recorded experimental results to PNG using Pillow."""
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.load_default(size=20)
SMALL = ImageFont.load_default(size=15)

def bars(title, labels, values, filename, bound=None):
    img=Image.new('RGB',(1100,650),'white'); d=ImageDraw.Draw(img)
    d.text((55,25),title,font=FONT,fill='#172b4d')
    maximum=bound or max(values)*1.15 or 1
    left, right, top, bottom=270,1020,85,590
    gap=(bottom-top)/len(labels)
    for i,(label,value) in enumerate(zip(labels,values)):
        y=top+i*gap; width=(right-left)*value/maximum
        d.text((20,y+5),str(label)[:28],font=SMALL,fill='#202020')
        d.rectangle((left,y+5,left+width,y+gap*.7),fill='#3264a8')
        d.text((min(left+width+8,1020),y+5),f'{value:.3f}',font=SMALL,fill='#202020')
    img.save(OUT/filename)

def main():
    quality=pd.read_json(ROOT/'outputs/data_audit/data_quality.json',typ='series')
    bars('Dataset class counts',list(quality.diagnosis_counts),list(quality.diagnosis_counts.values()),'class_counts.png')
    board=pd.read_csv(ROOT/'outputs/corrected/leaderboard.csv')
    bars('Validation average precision by model',board.model.tolist(),board.pr_auc.tolist(),'model_comparison.png',1.1)
    imp=pd.read_csv(ROOT/'outputs/corrected/permutation_importance.csv').head(10)
    bars('Test permutation importance: average precision drop',imp.feature.tolist(),imp.mean_drop_pr_auc.tolist(),'importance.png')
    runs=pd.read_csv(ROOT/'outputs/robustness/repeated_splits.csv')
    bars('Test recall across 10 repeated splits',runs.seed.astype(str).tolist(),runs.recall.tolist(),'recall_splits.png',1.1)
    bars('Test flag rate across 10 repeated splits',runs.seed.astype(str).tolist(),runs.test_flag_rate.tolist(),'flag_splits.png',.4)
    print('Saved figures in',OUT)

if __name__=='__main__':main()
