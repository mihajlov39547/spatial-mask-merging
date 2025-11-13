from smm import SMMPrediction, SpatialMaskMerger

pred = SMMPrediction('test.jpg')
pred.add_annotation('car', 0, 0.9, (10,10,50,50), [[[10,10],[50,50]]])
merger = SpatialMaskMerger()
result = merger.merge(pred, (100,100))
print(f'✅ API works: {len(result)} cluster(s), Score: {result[0]["score"]:.2f}')
