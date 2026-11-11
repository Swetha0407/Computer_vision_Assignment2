from pathlib import Path
import sys, time, json, cv2, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from dataset_generator import main as generate
DATA=ROOT/"data"/"synthetic"; OUT=ROOT/"outputs"; RES=ROOT/"results"
OUT.mkdir(exist_ok=True); RES.mkdir(exist_ok=True); cv2.setNumThreads(1)
def timed(fn):
    t=time.perf_counter(); v=fn(); return v,(time.perf_counter()-t)*1000
def gray(im): return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
def stats(im):
    g=im if im.ndim==2 else gray(im)
    return float(g.std()),float(cv2.Laplacian(g,cv2.CV_64F).var()),float(g.mean())
def panel(path,ims):
    arr=[]
    for im in ims:
        if im.ndim==2: im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        arr.append(cv2.resize(im,(256,192)))
    cv2.imwrite(str(path),np.hstack(arr))
def fft_filter(g,kind,cut=25,n=2):
    f=np.fft.fftshift(np.fft.fft2(g.astype(np.float32)))
    h,w=g.shape; y,x=np.ogrid[:h,:w]; d=np.sqrt((x-w/2)**2+(y-h/2)**2)
    if kind=="ideal_lpf": H=(d<=cut)
    elif kind=="butter_lpf": H=1/(1+(d/(cut+1e-9))**(2*n))
    elif kind=="gaussian_lpf": H=np.exp(-(d*d)/(2*cut*cut))
    elif kind=="ideal_hpf": H=(d>cut)
    elif kind=="butter_hpf": H=1-1/(1+(d/(cut+1e-9))**(2*n))
    else: H=1-np.exp(-(d*d)/(2*cut*cut))
    return np.clip(np.abs(np.fft.ifft2(np.fft.ifftshift(f*H))),0,255).astype(np.uint8)
def prewitt(g):
    k=np.array([[-1,0,1],[-1,0,1],[-1,0,1]],np.float32)
    gx=cv2.filter2D(g,cv2.CV_32F,k); gy=cv2.filter2D(g,cv2.CV_32F,k.T)
    return cv2.convertScaleAbs(cv2.magnitude(gx,gy))
def region_grow(g,seed,th=18):
    h,w=g.shape; seen=np.zeros((h,w),np.uint8); out=np.zeros((h,w),np.uint8)
    sy,sx=seed; ref=int(g[sy,sx]); stack=[(sy,sx)]; seen[sy,sx]=1
    while stack:
        y,x=stack.pop(); out[y,x]=255
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and not seen[yy,xx]:
                seen[yy,xx]=1
                if abs(int(g[yy,xx])-ref)<=th: stack.append((yy,xx))
    return out
def region_grow(g,seed,th=18):
    h,w=g.shape; seen=np.zeros((h,w),np.uint8); out=np.zeros((h,w),np.uint8)
    sy,sx=seed; ref=int(g[sy,sx]); stack=[(sy,sx)]; seen[sy,sx]=1
    while stack:
        y,x=stack.pop(); out[y,x]=255
        for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
            if 0<=yy<h and 0<=xx<w and not seen[yy,xx]:
                seen[yy,xx]=1
                if abs(int(g[yy,xx])-ref)<=th: stack.append((yy,xx))
    return out
def main():
    generate()
    labels=pd.read_csv(DATA/"labels.csv")
    samples=[(r.filename,cv2.imread(str(DATA/r.filename)),r.depth_group) for r in labels.itertuples()]
    summary={"dataset":"30 deterministic synthetic scenes","working_resolution":"512x384","tasks":{}}
    # Task 1 grey-level transforms
    rows=[]
    for name,im,depth in samples:
        g=gray(im)
        a,ta=timed(lambda:cv2.normalize(g,None,0,255,cv2.NORM_MINMAX))
        b,tb=timed(lambda:np.clip(255*np.power(g.astype(np.float32)/255,0.75),0,255).astype(np.uint8))
        c,tc=timed(lambda:(255*np.log1p(g.astype(np.float32))/np.log(256)).astype(np.uint8))
        rows.append({"sample":name,"depth":depth,"linear_stretch_ms":ta,"gamma_ms":tb,"log_ms":tc,
                     "linear_contrast":stats(a)[0],"gamma_contrast":stats(b)[0],"log_contrast":stats(c)[0]})
    pd.DataFrame(rows).to_csv(RES/"task_01_grey_level.csv",index=False)
    panel(OUT/"task01_grey_level.png",[g,a,b,c]); summary["tasks"]["1"]={"samples":len(rows),"methods":["linear stretch","gamma","log"]}
    # Task 2 histogram processing
    shallow=gray(samples[0][1]); hist_ref=cv2.calcHist([shallow],[0],None,[256],[0,256]).ravel()
    rows=[]
    for name,im,depth in samples:
        g=gray(im)
        a,ta=timed(lambda:cv2.equalizeHist(g))
        b,tb=timed(lambda:cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(g))
        def match():
            h=cv2.calcHist([g],[0],None,[256],[0,256]).ravel()
            c1=np.cumsum(h)/h.sum(); c2=np.cumsum(hist_ref)/hist_ref.sum()
            lut=np.interp(c1,c2,np.arange(256)).clip(0,255).astype(np.uint8)
            return cv2.LUT(g,lut)
        c,tc=timed(match)
        rows.append({"sample":name,"depth":depth,"global_equalization_ms":ta,"clahe_ms":tb,"histogram_matching_ms":tc,
                     "global_contrast":stats(a)[0],"clahe_contrast":stats(b)[0],"matched_contrast":stats(c)[0]})
    pd.DataFrame(rows).to_csv(RES/"task_02_histogram.csv",index=False)
    panel(OUT/"task02_histogram.png",[g,a,b,c]); summary["tasks"]["2"]={"samples":len(rows),"methods":["global equalization","CLAHE","histogram matching"]}
    # Task 3 spatial smoothing
    rows=[]
    for name,im,depth in samples:
        g=gray(im)
        a,ta=timed(lambda:cv2.blur(g,(5,5)))
        b,tb=timed(lambda:cv2.medianBlur(g,5))
        c,tc=timed(lambda:cv2.bilateralFilter(g,7,45,45))
        rows.append({"sample":name,"depth":depth,"mean_ms":ta,"median_ms":tb,"bilateral_ms":tc,
                     "mean_contrast":stats(a)[0],"median_contrast":stats(b)[0],"bilateral_contrast":stats(c)[0],
                     "mean_edge_variance":stats(a)[1],"median_edge_variance":stats(b)[1],"bilateral_edge_variance":stats(c)[1]})
    pd.DataFrame(rows).to_csv(RES/"task_03_smoothing.csv",index=False)
    panel(OUT/"task03_smoothing.png",[g,a,b,c]); summary["tasks"]["3"]={"samples":len(rows),"methods":["mean","median","bilateral"]}
    # Task 4 sharpening
    rows=[]
    for name,im,depth in samples:
        g=gray(im)
        lap,tl=timed(lambda:cv2.addWeighted(g,1.0,cv2.Laplacian(g,cv2.CV_8U),-0.5,0))
        blur=cv2.GaussianBlur(g,(5,5),1.0)
        un,tu=timed(lambda:cv2.addWeighted(g,1.5,blur,-0.5,0))
        hb,th=timed(lambda:cv2.addWeighted(g,2.0,blur,-1.0,0))
        rows.append({"sample":name,"depth":depth,"laplacian_ms":tl,"unsharp_ms":tu,"highboost_ms":th,
                     "laplacian_edge_variance":stats(lap)[1],"unsharp_edge_variance":stats(un)[1],"highboost_edge_variance":stats(hb)[1]})
    pd.DataFrame(rows).to_csv(RES/"task_04_sharpening.csv",index=False)
    panel(OUT/"task04_sharpening.png",[g,lap,un,hb]); summary["tasks"]["4"]={"samples":len(rows),"methods":["Laplacian","unsharp masking","high-boost"]}
    # Task 5 frequency LPF
    rows=[]
    for name,im,depth in samples:
        g=gray(im); outputs={}
        for kind in ("ideal_lpf","butter_lpf","gaussian_lpf"):
            outputs[kind],tm=timed(lambda k=kind:fft_filter(g,k))
            rowkey=kind+"_ms"
            if "row" not in locals(): row={}
            row[rowkey]=tm
            row[kind+"_std"]=stats(outputs[kind])[0]
        row.update({"sample":name,"depth":depth}); rows.append(row)
    pd.DataFrame(rows).to_csv(RES/"task_05_frequency_lpf.csv",index=False)
    panel(OUT/"task05_frequency_lpf.png",[g,outputs["ideal_lpf"],outputs["butter_lpf"],outputs["gaussian_lpf"]])
    summary["tasks"]["5"]={"samples":len(rows),"methods":["ideal LPF","Butterworth LPF","Gaussian LPF"]}
    # Task 6 frequency HPF
    rows=[]
    for name,im,depth in samples:
        g=gray(im); row={"sample":name,"depth":depth}; outs={}
        for kind in ("ideal_hpf","butter_hpf","gaussian_hpf"):
            outs[kind],tm=timed(lambda k=kind:fft_filter(g,k))
            row[kind+"_ms"]=tm; row[kind+"_std"]=stats(outs[kind])[0]
        rows.append(row)
    pd.DataFrame(rows).to_csv(RES/"task_06_frequency_hpf.csv",index=False)
    panel(OUT/"task06_frequency_hpf.png",[g,outs["ideal_hpf"],outs["butter_hpf"],outs["gaussian_hpf"]])
    summary["tasks"]["6"]={"samples":len(rows),"methods":["ideal HPF","Butterworth HPF","Gaussian HPF"]}
    # Task 7 edges
    rows=[]
    for name,im,depth in samples:
        g=gray(im)
        a,ta=timed(lambda:cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3))
        b,tb=timed(lambda:prewitt(g))
        c,tc=timed(lambda:cv2.Canny(g,55,135))
        rows.append({"sample":name,"depth":depth,"sobel_ms":ta,"prewitt_ms":tb,"canny_ms":tc,
                     "sobel_response_pixels":int(np.count_nonzero(np.abs(a)>30)),
                     "prewitt_edge_pixels":int(cv2.countNonZero(b)),"canny_edge_pixels":int(cv2.countNonZero(c))})
    pd.DataFrame(rows).to_csv(RES/"task_07_edges.csv",index=False)
    panel(OUT/"task07_edges.png",[g,cv2.convertScaleAbs(a),b,c]); summary["tasks"]["7"]={"samples":len(rows),"methods":["Sobel","Prewitt","Canny"]}
    # Task 8 thresholding
    rows=[]
    for name,im,depth in samples:
        g=gray(im)
        a,ta=timed(lambda:cv2.threshold(g,150,255,cv2.THRESH_BINARY)[1])
        (_,b),tb=timed(lambda:cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU))
        c,tc=timed(lambda:cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5))
        rows.append({"sample":name,"depth":depth,"global_ms":ta,"otsu_ms":tb,"adaptive_ms":tc,
                     "global_foreground_pixels":int(cv2.countNonZero(a)),"otsu_foreground_pixels":int(cv2.countNonZero(b)),
                     "adaptive_foreground_pixels":int(cv2.countNonZero(c))})
    pd.DataFrame(rows).to_csv(RES/"task_08_thresholding.csv",index=False)
    panel(OUT/"task08_thresholding.png",[g,a,b,c]); summary["tasks"]["8"]={"samples":len(rows),"methods":["global threshold","Otsu","adaptive"]}
    # Task 9 region segmentation
    rows=[]
    for name,im,depth in samples:
        g=gray(im); seed=(g.shape[0]//2,g.shape[1]//2)
        rg,tr=timed(lambda:region_grow(g,seed,22))
        binary=cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        (ncc,cc,st,cent),tcc=timed(lambda:cv2.connectedComponentsWithStats(binary))
        _,tw=timed(lambda:cv2.watershed(im.copy(),cv2.connectedComponents(binary)[1].astype(np.int32)))
        markers=cv2.connectedComponents(binary)[1].astype(np.int32); ws=im.copy()
        cv2.watershed(ws,markers)
        rows.append({"sample":name,"depth":depth,"region_growing_ms":tr,"connected_components_ms":tcc,"watershed_ms":tw,
                     "region_growing_pixels":int(cv2.countNonZero(rg)),"connected_component_count":int(ncc-1),
                     "watershed_boundary_pixels":int(np.count_nonzero(markers==-1))})
    pd.DataFrame(rows).to_csv(RES/"task_09_region_segmentation.csv",index=False)
    panel(OUT/"task09_region_segmentation.png",[g,rg,cc.astype(np.uint8)*35,cv2.convertScaleAbs(markers)])
    summary["tasks"]["9"]={"samples":len(rows),"methods":["seeded region growing","connected components","watershed"]}
    # Task 10 edge linking/adaptation
    rows=[]
    for name,im,depth in samples:
        g=gray(im); edges=cv2.Canny(g,50,130)
        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        a,ta=timed(lambda:cv2.drawContours(np.zeros_like(g),contours,-1,255,1))
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        b,tb=timed(lambda:cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel,iterations=1))
        # Region-edge fusion: colour/brightness candidate mask intersected with dilated edge support.
        def fusion():
            hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            region=cv2.inRange(hsv,np.array([25,25,40]),np.array([105,255,230]))
            edge_support=cv2.dilate(edges,kernel,iterations=2)
            return cv2.bitwise_and(region,edge_support)
        c,tc=timed(fusion)
        rows.append({"sample":name,"depth":depth,"contour_chaining_ms":ta,"morphological_linking_ms":tb,"region_edge_fusion_ms":tc,
                     "contour_count":len(contours),"linked_edge_pixels":int(cv2.countNonZero(b)),
                     "fusion_pixels":int(cv2.countNonZero(c))})
    pd.DataFrame(rows).to_csv(RES/"task_10_edge_linking.csv",index=False)
    panel(OUT/"task10_edge_linking.png",[g,a,b,c])
    summary["tasks"]["10"]={"samples":len(rows),"methods":["contour chaining","morphological linking","region-edge fusion"]}
    summary["numerical_answers"]={
      "task1_linear_stretch":round((110-70)*255/(210-70),4),
      "task2_cumulative_probability_through_level3":round((12+20+30+40)/202,6),
      "task3_mean":38,"task3_median":25,
      "task4_unsharp_output":100,
      "task5_gaussian_lpf_radius30_cutoff20":"relatively suppressed",
      "task6_complementary_hpf":0.7,
      "task7_gradient_magnitude":float(np.hypot(40,50)),
      "task8_precision":0.75,"task8_recall":0.833333,
      "task9_total_area_pixels":550,
      "task10_boundary_link_success_rate":0.85
    }
    (RES/"run_summary.json").write_text(json.dumps(summary,indent=2))
    print("Completed all 10 tasks on",len(samples),"synthetic samples.")
    print("Generated",len(list(RES.glob("*.csv"))),"CSV files and",len(list(OUT.glob("*.png"))),"visual outputs.")
if __name__=="__main__": main()
