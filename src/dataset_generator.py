from pathlib import Path
import cv2, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"synthetic"
DATA.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(20261112); W,H=512,384
def make_scene(i):
    depth=i%3
    sand=RNG.normal(132-8*depth,13+2*depth,(H,W))
    im=np.zeros((H,W,3),np.uint8)
    im[:,:,0]=np.clip(sand*.78+18+depth*8,0,255)
    im[:,:,1]=np.clip(sand*.93+10+depth*5,0,255)
    im[:,:,2]=np.clip(sand*1.02-depth*3,0,255)
    coral=np.zeros((H,W),np.uint8); cx=int(RNG.integers(150,360)); cy=int(RNG.integers(145,260))
    for b in range(7):
        x=cx+int(RNG.integers(-45,46)); y=cy+int(RNG.integers(-20,35))
        ex=x+int(RNG.integers(-80,81)); ey=y-int(RNG.integers(65,145)); thick=int(RNG.integers(5,13))
        cv2.line(coral,(x,y),(ex,ey),255,thick,cv2.LINE_AA)
        for t in (.35,.62):
            bx=int(x+(ex-x)*t); by=int(y+(ey-y)*t)
            cv2.line(coral,(bx,by),(bx+int(RNG.integers(-40,41)),by-int(RNG.integers(20,65))),255,max(3,thick-3),cv2.LINE_AA)
    coral=cv2.GaussianBlur(coral,(3,3),0)
    col=(int(RNG.integers(105,145)),int(RNG.integers(135,185)),int(RNG.integers(175,220)))
    for c in range(3): im[:,:,c]=np.where(coral>0,np.clip(col[c]*(coral/255)+im[:,:,c]*(1-coral/255),0,255),im[:,:,c])
    damage=np.zeros((H,W),np.uint8)
    for _ in range(3):
        cv2.ellipse(damage,(int(RNG.integers(120,390)),int(RNG.integers(95,300))),
                    (int(RNG.integers(8,17)),int(RNG.integers(5,12))),int(RNG.integers(0,180)),0,360,255,-1)
    damage=cv2.bitwise_and(damage,coral)
    pale=np.zeros_like(im); pale[:]=(210,220,230)
    im=np.where(damage[:,:,None]>0,np.clip(im*.25+pale*.75,0,255),im).astype(np.uint8)
    algae=np.zeros((H,W),np.uint8)
    for _ in range(5):
        cv2.ellipse(algae,(int(RNG.integers(40,470)),int(RNG.integers(70,330))),
                    (int(RNG.integers(10,28)),int(RNG.integers(5,16))),int(RNG.integers(0,180)),0,360,255,-1)
    algae=cv2.GaussianBlur(algae,(7,7),2); green=np.zeros_like(im); green[:]=(75,155,105)
    im=np.where(algae[:,:,None]>0,np.clip(im*.55+green*.45,0,255),im).astype(np.uint8)
    haze=.10+.08*depth; water=np.zeros_like(im); water[:]=(150,175,185)
    im=cv2.addWeighted(im,1-haze,water,haze,0)
    im=np.clip(im.astype(float)+RNG.normal(0,2.5+depth*1.5,im.shape),0,255).astype(np.uint8)
    if i%4==0:
        for _ in range(80): cv2.circle(im,(int(RNG.integers(W)),int(RNG.integers(H))),1,(220,230,235),-1)
    return im,{"filename":f"coral_{i+1:03d}.png","depth_group":["shallow","mid","deep"][depth],
               "haze_level":round(haze,3),"damage_patch_count":3,"algae_distractors":5,
               "synthetic_coral_pixels":int(cv2.countNonZero(coral))}
def main():
    rows=[]
    for i in range(30):
        im,m=make_scene(i); cv2.imwrite(str(DATA/m["filename"]),im); rows.append(m)
    pd.DataFrame(rows).to_csv(DATA/"labels.csv",index=False)
    print("Generated 30 synthetic underwater coral scenes.")
if __name__=="__main__": main()
