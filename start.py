"""Cross-platform CV day/block notebook selector; self-contained student entry point."""
import argparse
import subprocess
import sys
from pathlib import Path
NOTEBOOKS={
    '1':'day1-block-01-clustering/01_clustering.py',
    '2':'day1-block-02-reduction/02_reduction.py',
    '3':'day1-block-03-segmentation/03_segmentation.py',
    '5':'day1-block-05-neural/05_neural.py',
}
DAY2_NOTEBOOKS={
    '1':'day2-block-01-cnns-vgg/01_cnns_vgg.py',
    '2':'day2-block-02-model-improvement/02_model_improvement.py',
    '3':'day2-block-03-transfer/03_transfer.py',
    '4':'day2-block-04-autoencoders/04_autoencoders.py',
    '5':'day2-block-05-detection/05_detection.py',
}

DAY3_NOTEBOOKS={
    '1':'day3-block-01-segmentation/01_segmentation.py',
    '2':'day3-block-02-siamese/02_siamese.py',
    '4':'day3-block-04-prompts/04_prompts.py',
    '5':'day3-block-05-deployment/05_deployment.py',
}

def main():
    p=argparse.ArgumentParser(description='Open a prepared CV marimo practice by day and block.')
    p.add_argument('--day',choices=['1','2','3'],default='1')
    p.add_argument('--block',default='1');p.add_argument('--check',action='store_true');p.add_argument('--port',type=int,default=2720)
    p.add_argument('--offline-assets',action='store_true',help='Reject missing detector/foundation cache instead of downloading.')
    args=p.parse_args();root=Path(__file__).resolve().parent;notebooks={'1':NOTEBOOKS,'2':DAY2_NOTEBOOKS,'3':DAY3_NOTEBOOKS}[args.day]
    if args.block not in notebooks:p.error(f'Day {args.day} provides blocks: '+', '.join(notebooks))
    import marimo
    from cv_exercises.data import images,geometry,scene,classification_split
    x,y,_=images();assert x.shape==(300,64) and y.shape==(300,)
    assert (root/'exercises'/notebooks[args.block]).is_file()
    if args.day=='1':
        if args.block=='1':assert geometry('compact').shape==(150,2)
        if args.block=='2':
            import umap
            import scipy
        if args.block=='3':assert scene().shape==(32,48,3)
        if args.block=='5':assert len(classification_split()[3])==180
    elif args.day=='2':
        from cv_exercises import day2
        import json
        required={
            '1':['tiny-cnn.pt','vgg16-first-two-blocks.pt','vgg16-full-shape-trace.json'],
            '2':['tiny-cnn.pt','longer-training-reference.json'],
            '3':['tiny-cnn.pt','cached-features.npz'],
            '4':['ae-clean-4.pt','ae-clean-16.pt','ae-denoise-4.pt','ae-denoise-16.pt'],
            '5':['bus.jpg','zidane.jpg','detections.json'],
        }
        for name in required[args.block]:
            if not (day2.ASSETS/name).is_file():raise SystemExit(f'Missing bundled asset: {name}. Use the complete student package.')
        if args.block in ['1','2','3']:day2.tiny_model()
        if args.block=='4':day2.ae_inputs(0,4,.25)
        if args.block=='5':
            day2.prepare_detectors(download=not args.offline_assets)
            assert len(day2.detector_snapshot()['images'])==2
    else:
        from cv_exercises import day3
        required={'1':['unet-semantic.pt','unet-centers.pt','segmentation.npz'],'2':['siamese.pt','pairs.npz'],'4':['foundation-encodings.pt','bus.jpg','zidane.jpg','model-manifest.json'],'5':[]}
        for name in required[args.block]:
            if not (day3.ASSETS/name).is_file():raise SystemExit(f'Missing Day3 asset: {name}; use the complete student ZIP.')
        if args.block=='1':day3.segmentation(1)
        if args.block=='2':day3.pairs()
        if args.block=='4':
            day3.prepare_foundation(online=not args.offline_assets)
            assert len(day3.text_scores(['a photo of a bus','a photo of a person'])['scores'])==2
            assert day3.spatial_mask({'points':[[280,650]],'labels':[1]})['mask'].ndim==2
        if args.block=='5':day3.day2.tiny_model()
    print(f'Ready: CV Day {args.day} Block {args.block}, Python {sys.version.split()[0]}, marimo {marimo.__version__}. CPU only; bundled data.')
    if args.check:return 0
    print(f'Open http://127.0.0.1:{args.port}; Ctrl+C stops the editor. Save with Ctrl+S (Command+S on macOS).',flush=True)
    try:return subprocess.call([sys.executable,'-m','marimo','edit',str(root/'exercises'/notebooks[args.block]),'--host','127.0.0.1','--port',str(args.port),'--no-token','--no-sandbox','--skip-update-check'],cwd=root)
    except KeyboardInterrupt:return 0
if __name__=='__main__':raise SystemExit(main())
